---
type: "Runbook"
title: 混合精度 scales 用 TilingData 标志双绑定
description: scales 与主 dtype 不一致（bf16 主路径 + fp32 scales）时的完整实现模式：Tiling 检测写 TilingData 运行时标志（不扩 TilingKey 维度）、kernel 双 GlobalTensor 按标志绑定、标量读取分支免 Cast 直读 fp32——按主 dtype 单一绑定 scales 会静默读出错误数值。
tags: [task.implementation, topic.precision, topic.mixture_of_experts]
status: stable
sources: []
created_at: '2026-09-16T11:05:00Z'
updated_at: '2026-09-16T11:05:00Z'
platforms: ["950"]
---

# 混合精度 scales 用 TilingData 标志双绑定

## 现象

BenchNoHarnessMoeFinalizeRouting 算子（MoE 路由 finalize）需要支持混合精度场景：主路径张量（expanded_permuted_rows/skip1/skip2/bias/out）为 bf16，而 scales 为 fp32。在此之前的 ST 中这类用例被显式跳过（日志原文 `[SKIP] 混合精度路径（bf16+fp32 scales）迭代三实现，Real 模式跳过`），因为 kernel 模板只按主 dtype `D_T_MAIN` 绑定 scales 的 GlobalTensor——scales 实际是 fp32 时按 bf16 读会得到错误数值。全功能实现阶段的任务是在不破坏既有 12 种 TilingKey（3 dtype × 4 drop_pad_mode）二进制布局的前提下让 kernel 正确读取 fp32 scales，并让原先跳过的 3 个混合精度用例真正执行且通过。

## 适用场景

- 工程形态：算子仓开发，aclnn + 自定义 AscendC kernel（模板 kernel，`ASCENDC_TPL_SEL_PARAM` 双参数 TilingKey），C++ ST Real 模式 + UT（op_host Tiling 单测）验证。
- API 路径：`gert::TilingContext::GetInputDesc(5)->GetDataType()`（tiling 侧 dtype 检测）、TilingData 标志字段、`GlobalTensor<T>::SetGlobalBuffer`、`DataCopyPad`/`DataCopyPadExtParams`、`Cast(low→float, CAST_NONE)`、`Axpy`、`PipeBarrier<PIPE_ALL>`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35/DAV_3510）+ CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（带"可选/独立 dtype 缩放因子输入"的模板 kernel 均可套用）。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：fp16 主路径 + fp32 scales（本算子 Tiling 检测写死 bf16+fp32 组合，fp16+fp32 会被当作同 dtype 处理，语义未定义未验证）；scales 参与向量级乘法而非标量读取的场景。

## 解决配方

四件套实现（完整代码见下节）：

1. **Tiling 检测**：`GetInputMetaInfo` 同时取主 dtype 与 scales dtype，`isMixedPrecision = (mainDtype == DT_BF16 && hasScales && scalesDtype == DT_FLOAT)`，写入 `tiling->isMixedPrecision`。
2. **Kernel 双 GlobalTensor**：成员并存 `GlobalTensor<D_T_MAIN> scalesGM_`（同 dtype 路径）与 `GlobalTensor<float> scalesFp32GM_`（混合精度路径），`Init` 读 TilingData 标志后按标志 `SetGlobalBuffer`。
3. **标量读取三分支** `ReadScaleScalar(offsetElems)`：混合精度路径直接 `DataCopyPad` 4 字节 fp32 + `GetValue`（无需 Cast）；同 dtype fp32 路径同样直读；同 dtype 低精度路径 `DataCopyPad` + `Cast(low→float, CAST_NONE)` + `GetValue`。
4. **累加调用点**：`float scaleVal = ReadScaleScalar(i * topK_ + k)` 后 `Axpy(outBufFp32, gatherBufFp32, scaleVal, count)`。

配套 UT：TilingTestParam 增加 `scalesDtype`/`expectMixedPrecision` 字段，`BuildInputDescs` 用 scalesDtype 建 scales 描述，断言 `isMixedPrecision` 与 TilingKey（仍按主 dtype 编码）。

## 完整代码

Tiling 侧检测与传递（op_host/arch35/*_tiling.cpp，GetInputMetaInfo 内）：

```cpp
    // 输入 dtype: 主 dtype 与 scales dtype 分别获取
    auto epDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epDesc);
    info.mainDtype = epDesc->GetDataType();
    if (info.hasScales) {
        auto scalesDesc = context->GetInputDesc(5);
        OP_CHECK_NULL_WITH_CONTEXT(context, scalesDesc);
        info.scalesDtype = scalesDesc->GetDataType();
    }
    // 混合精度: bf16 主路径 + fp32 scales
    info.isMixedPrecision = (info.mainDtype == ge::DT_BF16 && info.hasScales &&
                             info.scalesDtype == ge::DT_FLOAT);
```

```cpp
    // 6. 可选输入标志
    tiling->hasSkip1 = info.hasSkip1 ? 1 : 0;
    tiling->hasSkip2 = info.hasSkip2 ? 1 : 0;
    tiling->hasBias = info.hasBias ? 1 : 0;
    tiling->hasScales = info.hasScales ? 1 : 0;
    tiling->hasEfsr = info.hasEfsr ? 1 : 0;
    tiling->isMixedPrecision = info.isMixedPrecision ? 1 : 0;
```

Kernel 成员声明（op_kernel/arch35/bench_no_harness_moe_finalize_routing.h）：

```cpp
    // ===== GlobalTensor =====
    GlobalTensor<D_T_MAIN> expandedPermutedGM_;
    GlobalTensor<int32_t>  expandedSrcToDstGM_;
    GlobalTensor<D_T_MAIN> skip1GM_;
    GlobalTensor<D_T_MAIN> skip2GM_;
    GlobalTensor<D_T_MAIN> biasGM_;
    GlobalTensor<D_T_MAIN> scalesGM_;        // 同 dtype 路径: scales 为 D_T_MAIN
    GlobalTensor<float>    scalesFp32GM_;    // 混合精度路径: scales 为 fp32
    GlobalTensor<int32_t>  efsrGM_;
    GlobalTensor<D_T_MAIN> outGM_;
```

```cpp
    // ===== 可选输入标志 =====
    bool hasSkip1_ = false;
    bool hasSkip2_ = false;
    bool hasBias_ = false;
    bool hasScales_ = false;
    bool hasEfsr_ = false;
    bool isMixedPrecision_ = false;  // bf16 主路径 + fp32 scales
```

Init 中读标志与按标志绑定 buffer：

```cpp
    hasSkip1_ = (tilingData->hasSkip1 != 0);
    hasSkip2_ = (tilingData->hasSkip2 != 0);
    hasBias_ = (tilingData->hasBias != 0);
    hasScales_ = (tilingData->hasScales != 0);
    hasEfsr_ = (tilingData->hasEfsr != 0);
    isMixedPrecision_ = (tilingData->isMixedPrecision != 0);
```

```cpp
    if (hasScales_) {
        // 混合精度路径: scales 为 fp32, 使用 scalesFp32GM_
        // 同 dtype 路径: scales 为 D_T_MAIN, 使用 scalesGM_
        if (isMixedPrecision_) {
            scalesFp32GM_.SetGlobalBuffer((__gm__ float*)scales);
        } else {
            scalesGM_.SetGlobalBuffer((__gm__ D_T_MAIN*)scales);
        }
    }
```

标量读取完整函数（修复前 → 修复后；修复前签名收 `GlobalTensor<D_T_MAIN>&`，无法读 fp32 scales）：

```cpp
// 修复前（只支持同 dtype）：
template <typename D_T_MAIN, int DROP_PAD_MODE>
__aicore__ inline float BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE>::ReadScaleScalar(
    const GlobalTensor<D_T_MAIN>& gm, int64_t offsetElems)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(D_T_MAIN);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<D_T_MAIN> padParams{false, 0, 0, 0};

    if constexpr (std::is_same_v<D_T_MAIN, float>) {
        // fp32 路径: 直接 DataCopyPad + GetValue (无 Cast)
        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        DataCopyPad(scaleLocal, gm[offsetElems], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        return scaleLocal.GetValue(0);
    } else {
        // 低精度路径: DataCopyPad + Cast(low→float) + GetValue
        LocalTensor<D_T_MAIN> scaleLocal = scaleBuf_.Get<D_T_MAIN>();
        LocalTensor<float> scaleFp32 = scaleBufFp32_.Get<float>();
        DataCopyPad(scaleLocal, gm[offsetElems], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(scaleFp32, scaleLocal, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_ALL>();
        return scaleFp32.GetValue(0);
    }
}
```

```cpp
// 修复后（终版，三分支：混合精度 / fp32 / 低精度）：
template <typename D_T_MAIN, int DROP_PAD_MODE>
__aicore__ inline float BenchNoHarnessMoeFinalizeRouting<D_T_MAIN, DROP_PAD_MODE>::ReadScaleScalar(
    int64_t offsetElems)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    // 混合精度路径: scales 为 fp32, 直接 DataCopyPad + GetValue (无需 Cast)
    if (isMixedPrecision_) {
        copyParams.blockLen = sizeof(float);
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        DataCopyPad(scaleLocal, scalesFp32GM_[offsetElems], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        return scaleLocal.GetValue(0);
    }

    // 同 dtype 路径: scales 为 D_T_MAIN
    copyParams.blockLen = sizeof(D_T_MAIN);
    DataCopyPadExtParams<D_T_MAIN> padParams{false, 0, 0, 0};

    if constexpr (std::is_same_v<D_T_MAIN, float>) {
        // fp32 路径: 直接 DataCopyPad + GetValue (无 Cast)
        LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
        DataCopyPad(scaleLocal, scalesGM_[offsetElems], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        return scaleLocal.GetValue(0);
    } else {
        // 低精度路径: DataCopyPad + Cast(low→float) + GetValue
        LocalTensor<D_T_MAIN> scaleLocal = scaleBuf_.Get<D_T_MAIN>();
        LocalTensor<float> scaleFp32 = scaleBufFp32_.Get<float>();
        DataCopyPad(scaleLocal, scalesGM_[offsetElems], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(scaleFp32, scaleLocal, RoundMode::CAST_NONE, 1);
        PipeBarrier<PIPE_ALL>();
        return scaleFp32.GetValue(0);
    }
}
```

累加调用点（AccumulateExpert 内，不再传 GM 参数）：

```cpp
    // 6. scales 路径: Axpy outBufFp32 += scale * gatherBufFp32
    LocalTensor<float> outBufFp32 = outBufFp32_.Get<float>();
    if (hasScales_) {
        // 标量读取 scales[i, k] → scaleVal (转 float)
        // 混合精度路径: scales 为 fp32 直接读取; 同 dtype 路径: scales 为 D_T_MAIN 转 float
        float scaleVal = ReadScaleScalar(i * topK_ + k);
        // Axpy: outBufFp32 += scaleVal * gatherBufFp32
        Axpy(outBufFp32, gatherBufFp32, scaleVal, static_cast<int32_t>(count));
        PipeBarrier<PIPE_ALL>();
    } else {
        // 无 scales: outBufFp32 += gatherBufFp32 (scale=1.0)
        Add(outBufFp32, outBufFp32, gatherBufFp32, count);
        PipeBarrier<PIPE_ALL>();
    }
```

配套 UT 用例（tests/ut/op_host/test_*_tiling.cpp，混合精度组）：

```cpp
    // ===== Group 5: 混合精度 (bf16 主路径 + fp32 scales, 迭代三) =====
    // bf16 + fp32 scales → isMixedPrecision=1, TilingKey 仍按主 dtype (bf16) 编码
    // mode 0: TK_8 (key=27)
    {"mixed_precision_bf16_fp32_mode0",
     ge::DT_BF16, 4, 2, 64, 8, 0,
     {true, true, true, true, true},
     ge::DT_FLOAT,  // scalesDtype = fp32 (混合精度)
     ge::GRAPH_SUCCESS, ComputeExpectTilingKey(ge::DT_BF16, 0), 1, 4,
     ComputeExpectHAlignElements(ge::DT_BF16), ComputeExpectExpertCapacity(0),
     true},  // expectMixedPrecision = true
```

## 设计决策

- **不新增 TilingKey，用 TilingData 运行时标志**：TilingKey 保持（主 dtype × drop_pad_mode）12 种编码；若为混合精度扩一维 TilingKey，模板实例/二进制从 12 翻到 24，收益只是省一次 bool 分支，成本是打包体积与维护复杂度，放弃。UT 明确断言"bf16 + fp32 scales → isMixedPrecision=1，TilingKey 仍按主 dtype (bf16) 编码"以锁定该决策。
- **kernel 双 GlobalTensor 按标志绑定，而非把 scales dtype 模板化**：保持 kernel 模板签名 `template <typename D_T_MAIN, int DROP_PAD_MODE>` 不变，scales 的 dtype 差异收敛到一个成员标志 + 两个 buffer，改动面最小。
- **混合精度路径免 Cast 直读 fp32 标量**：scales 本就是 fp32，无需走低精度路径的 `Cast(low→float, CAST_NONE)`，省一次转换与同步。
- 替代方案（scales 模板参数化）未实现。

## 验证信号

来源说明：全功能实现后的一轮完整验证：算子包重新编译安装 → UT → ST（Real 模式 NPU）。

- ST 混合精度 3 用例由 SKIP 转 PASS：MERE=7.35e-08、MARE=6.13e-03（threshold=7.81e-03，524288 elems，small_value: 2992 elems，max_abs_err=1.53e-05）；另两用例 MERE=4.38e-08/MARE=7.46e-03、MERE=1.26e-07/MARE=7.41e-03（来源平台：950；证据等级：tool_observed）
- UT 31/31 通过，其中 3 个新增混合精度用例断言 isMixedPrecision=1 且 TilingKey 仍按 bf16 编码（来源平台：950；证据等级：tool_observed）
- 全量 ST 57/57（4 L0 + 53 L1）0 失败 0 跳过，此前各阶段用例（12 TilingKey、可选输入组合、边界用例）无回归（来源平台：950；证据等级：tool_observed）
- PyTorch 接入路径复验混合精度用例 PASS：MERE=7.20e-08、MARE=7.75e-03（threshold=7.81e-03）（来源平台：950；证据等级：tool_observed）

## 失效边界

- 混合精度组合仅实现并验证 bf16 主路径 + fp32 scales（Tiling 检测写死该组合）；fp16 主路径 + fp32 scales 会被当作同 dtype fp16 处理，语义未定义、未验证。
- `ReadScaleScalar` 为逐 (i,k) 标量 `DataCopyPad`；K 很大时的性能代价未评估（无 profiling 信号，本卡不含性能结论）。
- 仅在 Ascend950PR + CANN 9.1.0 验证；其他 arch 的 buffer/同步语义未验证（未验证，推断机制可迁移）。
- 混合精度用例 MARE（6~7.5e-3）接近 bf16 阈值 7.81e-3，误差余量小；换数据分布或更长累加链可能超阈，未做敏感性分析。

**未决项**：

- fp16 主路径 + fp32 scales 组合的支持/拒绝语义未定义（Tiling 检测只认 bf16）。
## 坏实践

- 按主 dtype 单一 GlobalTensor 绑定所有输入——scales 实际 dtype 不同时读出错误数值且不报错（这正是混合精度用例此前必须 SKIP 的原因）。
- 为混合精度扩 TilingKey 维度——二进制数量翻倍，且该信息用 TilingData 运行时标志即可承载。

## 误诊

- 无（本卡为实现模式，非缺陷定位；实现后首轮全量即通过）。
