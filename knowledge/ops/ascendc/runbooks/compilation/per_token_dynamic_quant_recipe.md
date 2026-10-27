---
type: "Runbook"
title: per-token 动态量化 arch35 kernel 完整配方
description: Ascend950PR（arch35）per-token 对称动态量化算子完整实现配方：单 token 全载多核切分、fp32 计算链（Cast→Abs→ReduceMax→Muls→Round→clamp）、两步 Cast 落 int8、DataCopyPad 非对齐搬运、sharedTmpBuffer 双视图；UT 12/12 + ST 真机 5/5 验证。
tags: [task.implementation, topic.compute.quantization, topic.compute.reduction, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# per-token 动态量化 arch35 kernel 完整配方

## 现象（任务场景）

在 Ascend950PR（DAV_3510/arch35，CANN 9.1.0）上从零实现自定义算子 BenchNoHarnessDynamicQuant 的单 TilingKey 骨架（fp16 输入）：per-token 对称动态量化沿 last-dim——

- `scale = clamp(max(|x_token|), 1e-12) / 127`（fp32，shape = x.shape[:-1]）
- `y = clamp(round(x / scale), -128, 127)`（int8，shape = x.shape）

golden 为 torch 的 `nearbyint`（half-to-even）+ clamp 防除零/防截断；精度标准 y |actual-golden| ≤ 1.0、scale MERE < 2^-13 / MARE < 10×2^-13。要求编译通过、Kernel 二进制生成、op_host UT 通过、ST 真机 5 条 L0 用例全过。本卡沉淀该算子骨架阶段最终验证通过的完整实现（三轮编译修复 + 一次精度修复后的终态）。

## 适用场景

- 工程形态：AscendC 自定义算子（ops-registry-invoke 结构：op_host tiling + op_kernel arch35 + op_api aclnn 两段式）。
- API 路径：TQue/TBuf 流水（AllocTensor/EnQue/DeQue/FreeTensor）、DataCopyPad（非对齐搬运）、Cast/Abs/ReduceMax/Muls/Round/Mins/Maxs Vector 链、GetBlockNum/GetBlockIdx 多核切分、ASCENDC_TPL_SEL_PARAM 模板选择。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510/arch35）+ CANN 9.1.0，x86_64 宿主编译（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——per-token/per-axis 量化类算子（absMax→scale→除法→取整→clamp→int8）在 arch35 上的通用骨架；非量化类的 reduce+elementwise 混合算子也可参考其 buffer/流水组织。
- 路线优先级：recommended_fastpath（单 TilingKey 骨架路线）。
- 明确不适用 / 未验证场景：bf16 输入分支（T_IN=bfloat16_t，后续迭代范围，未实测）；lastDim 超 UB 单 token 装不下的大尾维（需 AR-ColSplit 类切分，本卡未涉及）；多 token 批处理（tileTokens>1，后续迭代范围）。

## 设计决策

1. **单 token 全载（AR-FullLoad）**：本阶段固定 `tileTokens=1`，每 token 一次 CopyIn→Compute→CopyOut；lastDim ∈ [1,16384] 时 fp32 工作区（lastDimAlign×4B）+ 输入/输出 buffer 均可放进 248KB UB。多 token 批处理推迟（收益是摊薄 per-token 同步开销，代价是 UB 峰值管理复杂度）。
2. **fp32 计算链**：fp16 原始数据先 `Cast` 到 fp32 再 Abs/ReduceMax/Muls/Round/clamp——absMax 与除法在 fp32 上做，避免 fp16 中间舍入累积；代价是 xFp32 工作区 4B/元素 + 原始数据要重 Cast 一次（Abs 是原地操作会破坏原始值，Step 5 显式重 Cast）。
3. **reduceTmpBuf 双视图共用**：同一 `TBuf` 以 `Get<float>()` 供 ReduceMax、以 `Get<uint8_t>()` 供 Round（两接口对 sharedTmpBuffer 的类型要求不同），串行使用零冲突；尺寸由 tiling 侧 `ComputeReduceBufSize`（perRepeat=256/typeSize、perBlock=32/typeSize，最小 4096B）给出。
4. **scaleLocal 临时复用**：ReduceMax 的 reduceDst 直接借用尚未写入有效数据的 scale 输出 LocalTensor，`GetValue(0)` 读 absMax 后 `SetValue(0, scale)` 覆写为最终 scale——省掉独立 reduceDst buffer；正确性依赖 V 管线内顺序（ReduceMax→GetValue→SetValue 同管程）与 EnQue 的 V→MTE3 同步。
5. **两步 Cast 落 int8**：`Cast` 在 Atlas A3 产品线不支持 float→int8_t（编译过但运行期输出错乱），终态用 fp32→half（CAST_ROUND）→int8_t（CAST_NONE），中间 half 借用输入 xLocal（原始输入此时已消费完）；[-128,127] 整数在 half 上无损表示，链路零精度损失（同类"向量 Cast 不支持宽跨 dtype、经中转类型两步走"的先例见[向量Cast不支持float转int64，经int32中转卡](vector_cast_float_to_int64_use_int32_pivot.md)）。
6. **多核切分自取核数**：kernel 内 `GetBlockNum()` 取总核数、`GetBlockIdx()` 取核号，token 均分 + `myStart_>=myEnd_` 早退（空任务核直接返回）；tiling 侧同时以 `PlatformAscendC::GetCoreNumAiv()` 上报 blockDim，两侧数值一致。
7. **非对齐搬运全用 DataCopyPad**：lastDim 非 32B 对齐（如 100）时 `lastDimAlign=CeilAlign(lastDim, 8)`（fp32 8 元素=32B），CopyIn/CopyOut 用 DataCopyPadExtParams 处理尾块，UB buffer 按 lastDimAlign 分配。
8. **Double Buffer**：inQueueX/outQueueY/outQueueScale 均 `BUFFER_NUM=2`，输入搬运与计算交叠。

## 系统信息

- 工具链：CANN 9.1.0（`ASCEND_HOME_PATH` 指向 CANN 安装根），CMake 3.20.5 + G++ 13.3.0（x86_64 宿主），构建入口 `bash build.sh --soc=ascend950 -j8`。
- 目标平台：Ascend950PR（DAV_3510/arch35，UB 248KB），SoC 名 ascend950。
- 目录布局（算子工程内）：`op_host/`（def/infershape/arch35 tiling）、`op_kernel/`（arch35 tiling_data.h、tiling_key.h、kernel .h + 入口 .cpp）、`op_api/`（L0 + aclnn L2）、`op_graph/`（proto）、`tests/ut/`（op_host UT）、`tests/st/`（C++ ST）。
- 相关文件清单（与本卡代码对应）：`op_kernel/arch35/bench_no_harness_dynamic_quant.h`（kernel 类，终态）、`op_host/arch35/bench_no_harness_dynamic_quant_tiling.cpp`（tiling）、`op_kernel/arch35/bench_no_harness_dynamic_quant_tiling_data.h`（TilingData 字段：totalTokens/lastDim/lastDimAlign/tileTokens/reduceTmpSize）。

## 完整代码

Kernel 类终态（op_kernel/arch35/bench_no_harness_dynamic_quant.h；由开发会话终版读取快照 + 两步 Cast 修复编辑拼装，verbatim）：

```cpp
/*!
 * \file bench_no_harness_dynamic_quant.h
 * \brief BenchNoHarnessDynamicQuant 算子 kernel 类定义（arch35 架构）
 *
 * 模板参数说明：
 *   - T_IN: 输入数据类型（half / bfloat16_t）
 *   - SCH_MODE: 调度模式（0=SINGLE_TOKEN, 1=MULTI_TOKEN）
 *
 * 单 Token 模式（SCH_MODE=0）：
 *   每个 tile 处理 1 个 token，CopyIn→Compute→CopyOut 流水
 *   Double Buffer on inputBuf + yOutBuf + scaleBuf
 *
 * 计算流程（per token）：
 *   Cast(fp16→fp32) → Abs → ReduceMax → 标量(clamp/div)
 *   → Cast(重载原始数据) → Muls(×invScale) → Round → Mins(127) → Maxs(-128)
 *   → Cast(fp32→int8)
 */

#ifndef BENCH_NO_HARNESS_DYNAMIC_QUANT_H
#define BENCH_NO_HARNESS_DYNAMIC_QUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bench_no_harness_dynamic_quant_tiling_data.h"
#include "bench_no_harness_dynamic_quant_tiling_key.h"

namespace NsBenchNoHarnessDynamicQuant {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;       // Double Buffer
constexpr float SCALE_DIVISOR = 127.0f;
constexpr float CLAMP_MIN_ABSMAX = 1e-12f;
constexpr float CLAMP_MAX_INT8 = 127.0f;
constexpr float CLAMP_MIN_INT8 = -128.0f;
constexpr uint32_t SCALE_BUF_BYTES = 32;  // 1 float, 32B aligned

template <typename T_IN, int SCH_MODE>
class KernelDynamicQuant {
public:
    __aicore__ inline KernelDynamicQuant(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                const BenchNoHarnessDynamicQuantTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t tokenIdx);
    __aicore__ inline void Compute(uint64_t tokenIdx);
    __aicore__ inline void CopyOut(uint64_t tokenIdx);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;       // 输入 double buffer
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;     // y 输出 double buffer
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueScale; // scale 输出 double buffer
    TBuf<TPosition::VECCALC> xFp32Buf;                   // fp32 工作区
    TBuf<TPosition::VECCALC> reduceTmpBuf;               // ReduceMax + Round 共用临时空间

    GlobalTensor<T_IN> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<float> scaleGm;

    uint64_t totalTokens_ = 0;
    uint32_t lastDim_ = 0;
    uint32_t lastDimAlign_ = 0;
    uint64_t myStart_ = 0;
    uint64_t myEnd_ = 0;
};

template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale,
    const BenchNoHarnessDynamicQuantTilingData* tiling)
{
    totalTokens_ = tiling->totalTokens;
    lastDim_ = tiling->lastDim;
    lastDimAlign_ = tiling->lastDimAlign;

    // 多核切分
    uint32_t blockDim = static_cast<uint32_t>(GetBlockNum());
    uint64_t tokensPerCore = (totalTokens_ + blockDim - 1) / blockDim;
    myStart_ = GetBlockIdx() * tokensPerCore;
    myEnd_ = (myStart_ + tokensPerCore > totalTokens_) ? totalTokens_ : (myStart_ + tokensPerCore);

    // GM 绑定
    xGm.SetGlobalBuffer((__gm__ T_IN*)x);
    yGm.SetGlobalBuffer((__gm__ int8_t*)y);
    scaleGm.SetGlobalBuffer((__gm__ float*)scale);

    // UB Buffer 分配
    uint32_t inBufSize = lastDimAlign_ * sizeof(T_IN);     // fp16: 2B per element
    uint32_t fp32BufSize = lastDimAlign_ * sizeof(float);  // fp32: 4B per element
    uint32_t outBufSize = lastDimAlign_ * sizeof(int8_t);  // int8: 1B per element

    pipe.InitBuffer(inQueueX, BUFFER_NUM, inBufSize);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, outBufSize);
    pipe.InitBuffer(outQueueScale, BUFFER_NUM, SCALE_BUF_BYTES);
    pipe.InitBuffer(xFp32Buf, fp32BufSize);
    pipe.InitBuffer(reduceTmpBuf, tiling->reduceTmpSize);
}

template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::CopyIn(uint64_t tokenIdx)
{
    LocalTensor<T_IN> xLocal = inQueueX.template AllocTensor<T_IN>();
    // 搬入 1 行 lastDim 个元素
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = lastDim_ * sizeof(T_IN);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<T_IN> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[tokenIdx * lastDim_], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::Compute(uint64_t tokenIdx)
{
    LocalTensor<T_IN> xLocal = inQueueX.template DeQue<T_IN>();
    LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
    LocalTensor<float> scaleLocal = outQueueScale.template AllocTensor<float>();
    LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
    LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
    LocalTensor<uint8_t> reduceTmpUint8 = reduceTmpBuf.Get<uint8_t>();

    // Step 1: Cast fp16/bf16 → fp32
    Cast(xFp32, xLocal, RoundMode::CAST_NONE, lastDim_);

    // Step 2: Abs (in-place)
    Abs(xFp32, xFp32, lastDim_);

    // Step 3: ReduceMax → absMax（Level 2 接口，需 sharedTmpBuffer）
    // 临时复用 scaleLocal 存放 reduce 结果（scaleLocal 还未写入有效数据）
    ReduceMax<float>(scaleLocal, xFp32, reduceTmp, static_cast<int32_t>(lastDim_), false);
    float absMax = scaleLocal.GetValue(0);

    // Step 4: 标量计算: clamp(1e-12) → /127 → scale → invScale
    float clampedMax = (absMax > CLAMP_MIN_ABSMAX) ? absMax : CLAMP_MIN_ABSMAX;
    float scale = clampedMax / SCALE_DIVISOR;
    float invScale = 1.0f / scale;

    // Step 5: 重新 Cast 原始数据到 fp32（Abs 已破坏原始数据）
    Cast(xFp32, xLocal, RoundMode::CAST_NONE, lastDim_);

    // Step 6: Muls × invScale (in-place) = x / scale
    Muls(xFp32, xFp32, invScale, lastDim_);

    // Step 7: Round (half-to-even, in-place, 需 sharedTmpBuffer<uint8_t>)
    Round(xFp32, xFp32, reduceTmpUint8, lastDim_);

    // Step 8: Clamp(-128, 127) = Mins(127) then Maxs(-128)
    Mins(xFp32, xFp32, CLAMP_MAX_INT8, lastDim_);
    Maxs(xFp32, xFp32, CLAMP_MIN_INT8, lastDim_);

    // Step 9: Cast fp32 → int8 (两步：float→half→int8, 因为 Cast 不支持 float→int8_t)
    //   9a: fp32 → half (CAST_ROUND, 值已在[-128,127]整数范围, half可精确表示)
    //   9b: half → int8_t (CAST_NONE, 值已是整数无需再舍入)
    Cast(xLocal, xFp32, RoundMode::CAST_ROUND, lastDim_);
    Cast(yLocal, xLocal, RoundMode::CAST_NONE, lastDim_);

    // 写入 scale 到 scaleLocal
    scaleLocal.SetValue(0, scale);

    outQueueY.EnQue<int8_t>(yLocal);
    outQueueScale.EnQue<float>(scaleLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::CopyOut(uint64_t tokenIdx)
{
    // 搬出 y
    LocalTensor<int8_t> yLocal = outQueueY.template DeQue<int8_t>();
    DataCopyExtParams yCopyParams;
    yCopyParams.blockCount = 1;
    yCopyParams.blockLen = lastDim_ * sizeof(int8_t);
    yCopyParams.srcStride = 0;
    yCopyParams.dstStride = 0;
    yCopyParams.rsv = 0;
    DataCopyPad(yGm[tokenIdx * lastDim_], yLocal, yCopyParams);
    outQueueY.FreeTensor(yLocal);

    // 搬出 scale (1 个 fp32)
    LocalTensor<float> scaleLocal = outQueueScale.template DeQue<float>();
    DataCopyExtParams scaleCopyParams;
    scaleCopyParams.blockCount = 1;
    scaleCopyParams.blockLen = sizeof(float);
    scaleCopyParams.srcStride = 0;
    scaleCopyParams.dstStride = 0;
    scaleCopyParams.rsv = 0;
    DataCopyPad(scaleGm[tokenIdx], scaleLocal, scaleCopyParams);
    outQueueScale.FreeTensor(scaleLocal);
}

template <typename T_IN, int SCH_MODE>
__aicore__ inline void KernelDynamicQuant<T_IN, SCH_MODE>::Process()
{
    if (myStart_ >= myEnd_) {
        return;
    }
    for (uint64_t tokenIdx = myStart_; tokenIdx < myEnd_; tokenIdx++) {
        CopyIn(tokenIdx);
        Compute(tokenIdx);
        CopyOut(tokenIdx);
    }
}

} // namespace NsBenchNoHarnessDynamicQuant
#endif // BENCH_NO_HARNESS_DYNAMIC_QUANT_H
```

Tiling 侧关键函数（op_host/arch35/bench_no_harness_dynamic_quant_tiling.cpp；sharedTmpBuffer 尺寸计算 + 平台核数 + 主 TilingFunc，verbatim）：

```cpp
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算 ReduceMax sharedTmpBuffer 大小
// 参考设计文档 §3.4 ComputeReduceBufSize
static uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize)
{
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;    // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, MIN_REDUCE_TMP_SIZE);
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BenchNoHarnessDynamicQuantTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2、获取输入 shape 和 dtype 信息
    auto inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    auto storageShape = inputShapeX->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2,
        OP_LOGE(context, "BenchNoHarnessDynamicQuant: input dim must be >= 2, got %zu", dimNum),
        return ge::GRAPH_FAILED);

    // 合轴：(A1, R) 二维矩阵
    // A1 = d0 * d1 * ... * d_{n-1} (token 总数)
    // R = d_n (last-dim)
    uint64_t totalTokens = 1;
    for (size_t i = 0; i < dimNum - 1; i++) {
        totalTokens *= static_cast<uint64_t>(storageShape.GetDim(i));
    }
    uint32_t lastDim = static_cast<uint32_t>(storageShape.GetDim(dimNum - 1));

    // last-dim 按 fp32 对齐（32B / 4B = 8 elements）
    uint32_t lastDimAlign = CeilAlign(static_cast<uint32_t>(lastDim), FP32_BLOCK_ELEMS);

    // 3、计算 reduceTmpSize
    uint32_t reduceTmpSize = ComputeReduceBufSize(lastDimAlign, sizeof(float));

    // 4、计算 tileTokens
    // 本阶段固定 SINGLE_TOKEN 模式（tileTokens=1）
    // 多 Token 批处理在后续迭代实现
    uint32_t tileTokens = 1;

    // 5、获取 WorkspaceSize
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 6、设置 TilingData
    BenchNoHarnessDynamicQuantTilingData* tiling = context->GetTilingData<BenchNoHarnessDynamicQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BenchNoHarnessDynamicQuantTilingData), 0, sizeof(BenchNoHarnessDynamicQuantTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalTokens = totalTokens;
    tiling->lastDim = lastDim;
    tiling->lastDimAlign = lastDimAlign;
    tiling->tileTokens = tileTokens;
    tiling->reduceTmpSize = reduceTmpSize;

    // 7、多核切分
    uint32_t blockDim = static_cast<uint32_t>(coreNum);
    context->SetBlockDim(blockDim);

    // 8、设置 TilingKey
    // 本阶段：仅 fp16 + SINGLE_TOKEN
    uint32_t dTypeX = static_cast<uint32_t>(ge::DT_FLOAT16);
    uint32_t schMode = 0;  // SINGLE_TOKEN
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, schMode);

    return ge::GRAPH_SUCCESS;
}
```

## 验证信号

来源说明：问题与验证来自 BenchNoHarnessDynamicQuant 开发与联调（Ascend950PR 真机、CANN 9.1.0），UT（op_host Tiling+InferShape，gtest）与 ST（C++ 工程，NPU 实测 vs CPU golden）。

- 编译：`Build completed successfully!`，Kernel 二进制 `BenchNoHarnessDynamicQuant_7cb48a37....o` 生成、custom_opp .run 包产出（来源平台：950；证据等级：tool_observed）。
- UT：12/12 通过（5 InferShape + 7 Tiling，含非对齐 lastDim、多维、[1,16384] 大尾维、1D 拒绝用例）（来源平台：950；证据等级：tool_observed）。
- ST 真机：5/5 条 L0 用例通过；y(int8) max_diff=0~1（标准 ≤1.0），scale(fp32) MERE=0、MARE=0（标准 MERE<2^-13、MARE<10×2^-13）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单 token 全载要求 `lastDimAlign×(sizeof(T_IN)+4+1) + reduceTmpSize` 可放进单核 UB；lastDim 上限 16384（本阶段实测上界），更大尾维需换切分策略（未验证）。
- bf16 输入分支未实测（推断 fp16 路径可迁移：Cast bf16→fp32 同为支持组合，但 Round/两步 Cast 的 bf16 行为未验证）。
- 两步 Cast 的无损性依赖值域先 clamp 到 [-128,127] 整数；通用 fp32→int8 场景（未 clamp）不适用。
- tileTokens 固定 1：本配方是骨架，多 token 批处理性能优化空间未开采（性能未评测，无 profiling 信号）。

**未决项**：

- 多 token 批处理（tileTokens>1）与 bf16 分支、多 TilingKey 组合的 key 值标定待后续迭代验证。
- 性能未评测（无 profiling 信号）：单 token 全载 + Double Buffer 的带宽利用率、以及 tail-core 负载均衡未测，优化方向未定。
## 坏实践

- Abs 原地操作后直接用同一 buffer 做量化——absMax 算完后原始数据已被破坏，必须重 Cast 原始输入（或双 buffer 分离）。
- 把 sharedTmpBuffer 统一按单一类型取视图——ReduceMax 与 Round 对 sharedTmpBuffer 的类型要求不同（float vs uint8_t），双视图串行共用即可，类型错配会出错。
