---
type: "Runbook"
title: "B16 交错 DINTLV 成对归零 — 转 fp32 后重排"
description: "B16 直接解交错经 Cast 后 lane 对不齐；先转连续 fp32 UB 再走 DINTLV_B32。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.data_movement.layout_transform"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["fp32_scratch"]
---

# B16 交错 DINTLV 成对归零 — 转 fp32 后重排

## 现象

在 Ascend950 Reg 路径实现交错旋转时，fp32 使用 `DINTLV_B32` 解交错、双路向量计算、`INTLV_B32` 重交错可与 CPU golden 完全一致，实测 `max_diff=0`；但把同一结构直接换成 B16 搬运并在寄存器中 Cast 后，fp16 样例出现 `max_diff=3.3447`、`mean=0.429179`，507127 个点误差超过 `1e-2`。

诊断输出显示第一对结果正确，后续位置出现成对的零和错位值；修改 chunk step 后失败数量完全不变，说明问题不在循环分块，而在 B16 解交错后的 lane 布局与后续 Cast/算术预期不一致。

## 适用场景

- 工程形态：kernel 直调 / Flash `.asc` 的 Reg SIMD 交错计算；
- API 路径：`LoadAlign<DIST_DINTLV_B16>`、`Reg::Cast`、`StoreAlign<DIST_INTLV_B16>`，以及 fp32-scratch 的 `DINTLV_B32/INTLV_B32`；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957b）/ CANN 9.0.1；
- 适用粒度：`operator_family`，适用于 B16 偶/奇 lane 成对计算与重排；
- 路线优先级：`safety_fallback`；这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）；
- 明确不适用：fp32 输入可直接走已验证的 B32 路径；其他 CANN 版本或平台的 B16 DINTLV lane 语义未验证。

## 根因

唯一根因是把 `DINTLV_B16` 产生的 B16 双寄存器 lane 布局直接当作逻辑偶/奇对，再经 `Reg::Cast` 做逐 lane 运算；该组合在实测工具链上没有保持后续算术所需的 `(x[2i], x[2i+1])` 配对，而连续 fp32 UB 上的 `DINTLV_B32` 能保持该配对。

## 解决配方

保留 fp32 输入的直接 B32 路线；对 fp16/bf16，把输入行与 cos/sin 分别 Cast 到连续 fp32 UB，在依赖 VF 之间插入 `PipeBarrier<PIPE_V>`，调用已经实测正确的 `DINTLV_B32/INTLV_B32`，最后 Cast 回 B16。

### before

故障状态的完整 `ComputeTile` 函数直接对模板 dtype 调用 interleave：

```cpp
__aicore__ inline void ComputeTile(uint32_t thisTile,
                                   const AscendC::LocalTensor<T> &cosUb,
                                   const AscendC::LocalTensor<T> &sinUb)
{
    auto inUb = inQueue_.DeQue<T>();
    auto outUb = outQueue_.AllocTensor<T>();
    for (uint32_t r = 0; r < thisTile; ++r) {
        auto inRow = inUb[static_cast<int64_t>(r) * D_];
        auto outRow = outUb[static_cast<int64_t>(r) * D_];
        if (mode_ == 1) {
            ComputeRopeInterleave<T>(outRow, inRow, cosUb, sinUb, D_);
        } else {
            ComputeRopeHalf<T>(outRow, inRow, cosUb, sinUb, D_);
        }
    }
    outQueue_.EnQue(outUb);
    inQueue_.FreeTensor(inUb);
}
```

### after

以下完整函数由轨迹的精确 Edit 内容与随后唯一的 `PIPE_V` 限定名修正重建，代码保持最终写入形态：

```cpp
__aicore__ inline void ComputeTile(uint32_t thisTile,
                                   const AscendC::LocalTensor<T> &cosUb,
                                   const AscendC::LocalTensor<T> &sinUb)
{
    auto inUb = inQueue_.DeQue<T>();
    auto outUb = outQueue_.AllocTensor<T>();
    if (mode_ == 1) { // interleave
        if constexpr (std::is_same<T, float>::value) {
            for (uint32_t r = 0; r < thisTile; ++r) {
                auto inRow = inUb[static_cast<int64_t>(r) * D_];
                auto outRow = outUb[static_cast<int64_t>(r) * D_];
                ComputeRopeInterleave<T>(outRow, inRow, cosUb, sinUb, D_);
            }
        } else {
            // b16: cast row+cos+sin to fp32, run the proven fp32 DINTLV_B32 interleave,
            // cast result back. PipeBarrier<PIPE_V> between V steps that pass data via UB.
            auto cosF = cosFp32Buf_.Get<float>();
            auto sinF = sinFp32Buf_.Get<float>();
            auto inF = inFp32Buf_.Get<float>();
            auto outF = outFp32Buf_.Get<float>();
            CastB16ToFp32<T>(cosF, cosUb, D_ / 2);
            CastB16ToFp32<T>(sinF, sinUb, D_ / 2);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < thisTile; ++r) {
                auto inRowT = inUb[static_cast<int64_t>(r) * D_];
                auto inRowF = inF[static_cast<int64_t>(r) * D_];
                auto outRowF = outF[static_cast<int64_t>(r) * D_];
                auto outRowT = outUb[static_cast<int64_t>(r) * D_];
                CastB16ToFp32<T>(inRowF, inRowT, D_);
                AscendC::PipeBarrier<PIPE_V>();
                ComputeRopeInterleaveFp32(outRowF, inRowF, cosF, sinF, D_);
                AscendC::PipeBarrier<PIPE_V>();
                CastFp32ToB16<T>(outRowT, outRowF, D_);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else { // half
        for (uint32_t r = 0; r < thisTile; ++r) {
            auto inRow = inUb[static_cast<int64_t>(r) * D_];
            auto outRow = outUb[static_cast<int64_t>(r) * D_];
            ComputeRopeHalf<T>(outRow, inRow, cosUb, sinUb, D_);
        }
    }
    outQueue_.EnQue(outUb);
    inQueue_.FreeTensor(inUb);
}
```

## 设计决策

- 选择 fp32-scratch，因为同一轨迹已用数值 probe 证明 `DINTLV_B32` 对逻辑偶/奇对完全正确；
- 放弃继续调 B16 chunk step：修改后仍是 57 passed、28 failed，说明它没有改变根因；
- fp32 路径保持直接 B32，不为已正确 dtype 付出额外 Cast 与 UB；
- 当前 B16 方案为 `safety_fallback`：优先保证正确性，尚无同轨迹、同 workload 的路线级性能 A/B。

## 完整代码

修复新增的 Cast VF、wrapper 和 fp32 interleave wrapper 均为轨迹中 verbatim 的完整函数：

```cpp
template <typename T>
__simd_vf__ inline void cast_b16_to_fp32_vf(__ubuf__ float *dstAddr, __ubuf__ T *srcAddr,
                                            uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    static constexpr AscendC::Reg::CastTrait kUp = {AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<T> t;
        AscendC::Reg::RegTensor<float> f;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(t, srcAddr + off);
        AscendC::Reg::Cast<float, T, kUp>(f, t, mask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + off, f, mask);
    }
}

template <typename T>
__simd_vf__ inline void cast_fp32_to_b16_vf(__ubuf__ T *dstAddr, __ubuf__ float *srcAddr,
                                            uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    static constexpr AscendC::Reg::CastTrait kDown = {AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t off = c * VL;
        uint32_t rem = count - off;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(rem);
        AscendC::Reg::RegTensor<float> f;
        AscendC::Reg::RegTensor<T> t;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(f, srcAddr + off);
        AscendC::Reg::Cast<T, float, kDown>(t, f, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + off, t, mask);
    }
}

template <typename T>
__aicore__ inline void CastB16ToFp32(const AscendC::LocalTensor<float> &dst,
                                     const AscendC::LocalTensor<T> &src, int64_t count)
{
    __ubuf__ float *dA = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ T *sA = (__ubuf__ T *)src.GetPhyAddr();
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t cc = (cnt + VL - 1) / VL;
    asc_vf_call<cast_b16_to_fp32_vf<T>>(dA, sA, cnt, cc);
}

template <typename T>
__aicore__ inline void CastFp32ToB16(const AscendC::LocalTensor<T> &dst,
                                     const AscendC::LocalTensor<float> &src, int64_t count)
{
    __ubuf__ T *dA = (__ubuf__ T *)dst.GetPhyAddr();
    __ubuf__ float *sA = (__ubuf__ float *)src.GetPhyAddr();
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t cnt = static_cast<uint32_t>(count);
    uint32_t cc = (cnt + VL - 1) / VL;
    asc_vf_call<cast_fp32_to_b16_vf<T>>(dA, sA, cnt, cc);
}

__aicore__ inline void ComputeRopeInterleaveFp32(const AscendC::LocalTensor<float> &out,
                                                 const AscendC::LocalTensor<float> &in,
                                                 const AscendC::LocalTensor<float> &cos,
                                                 const AscendC::LocalTensor<float> &sin,
                                                 int64_t d)
{
    __ubuf__ float *outA = (__ubuf__ float *)out.GetPhyAddr();
    __ubuf__ float *inA = (__ubuf__ float *)in.GetPhyAddr();
    __ubuf__ float *cosA = (__ubuf__ float *)cos.GetPhyAddr();
    __ubuf__ float *sinA = (__ubuf__ float *)sin.GetPhyAddr();
    constexpr uint32_t CHUNK_STEP = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t dCount = static_cast<uint32_t>(d);
    uint32_t dChunkCount = (dCount + CHUNK_STEP - 1) / CHUNK_STEP;
    asc_vf_call<rope_interleave_vf<float>>(outA, inA, cosA, sinA, dCount, dChunkCount);
}
```

## 性能影响

本轨迹只验证了最终整体算子的性能，没有对“直接 B16 路线”和“fp32-scratch 路线”做可比较的单路线 A/B；因此不得宣称该 fallback 性能最优。额外 UB、两次 Cast 和 VF barrier 都可能增加代价。

## 预防

- 新用 `DINTLV_B16/INTLV_B16` 时先做递增 lane 的最小化上板 probe，别从 B32 语义直接外推；
- 对成对算法打印前若干输出，交替零对比单纯 max_diff 更能揭示 lane mapping 问题；
- 改分块后失败集完全不变时停止继续调 chunk，回到数据布局和 Cast 链；
- 多 VF 经 UB 传递中间量时，在相邻步骤之间明确 `PipeBarrier<PIPE_V>`。

## 验证信号

来源为一次 Ascend950 Reg interleaved 路线的定向数值 probe、修复和最终评测：

- fp32 `DINTLV_B32/INTLV_B32` 路线 `max_diff=0`（950，证据等级: tool_observed）；
- 直接 B16 路线 `max_diff=3.3447`、`mean=0.429179`、507127 个点误差超过 `1e-2`，输出出现成对归零（950，证据等级: tool_observed）；
- fp32-scratch 修复后完整 NPU 套件 85/85 通过（950，证据等级: tool_observed）；
- 干净重建后仍为 85/85，cann-bench 20/20 case 通过（950，证据等级: tool_observed）。

## 失效边界

本卡证明的是 CANN 9.0.1、Ascend950 上该 B16 组合路线的实测行为，不声明其他版本的 `DINTLV_B16` 必然相同。fp32 输入无需 scratch；无法容纳 fp32 中间 UB 的极端 tile 必须先缩小 tile。

## 坏实践

- 因为 B32 解交错正确，就假定 B16 解交错经 Cast 后拥有完全相同的逻辑 lane 编号；
- 看到零值便改精度容差或输入分布；该故障是结构性 lane 错位，放宽容差无效。

## 误诊

最初把失败归为 dtype-aware chunk step；修改后仍是 57 passed、28 failed，工具结果否定了该假设。