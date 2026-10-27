---
type: "Runbook"
title: "B16 比较先升到 B32 锁定索引 lane"
description: "cummin/argmin 前缀扫描同时更新 B16 值与 int32 索引时，128-lane 比较 mask 不能驱动 64-lane 索引 Select；先升 float 统一 B32 lane。"
tags: ["task.debugging", "topic.precision", "topic.compute.elementwise.type_conversion", "topic.implementation.masking"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["prefix_scan", "lane_width", "common_b32_compute"]
---

# B16 比较先升到 B32 锁定索引 lane

## 现象

在 Ascend950 `operators/cummin/cummin.asc` 中，寄存器宽度和 CopyIn 行布局修正后，float32 与 int32 路径已通过，完整 NPU pytest 只剩 14 个失败，全部位于 float16/bfloat16 组合；其中 bfloat16 的 values 可接近参考值，但 int64 indices 仍出现多位置错配。

当时代码直接声明 `RegTensor<T>`，用 `Compare<T>` 生成 mask，再让同一个 mask 同时驱动 `Select<T>` 和 `Select<int32_t>`。活动元素数虽被限制为 64，但 mask 的元素宽度仍由 `T` 决定。

## 适用场景

- 工程形态：Ascend C Reg SIMD 内核同时更新一个 16 位 value 向量和一个 32 位 index/metadata 向量。
- API 路径：`Reg::Compare` 生成 `MaskReg`，随后由多个不同 dtype 的 `Reg::Select` 消费。
- 已验证环境：Ascend950PR_957c / `dav-3510`，CANN 9.0.1，注册平台 `950`。
- 适用粒度：`operator_family`，覆盖值为 half/bfloat16、伴随状态为 int32 的 lane-wise 选择链。
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请用另一路线（见 ## 相关）。本轨迹验证的是 32 位计算类型的正确性，未证明其性能最优。

## 根因

在 256B Reg 宽度下，B16 值寄存器有 128 个 lane，int32 索引寄存器只有 64 个 lane。`Compare<T>` 在 B16 语义下形成 128-lane mask，而 `Select<int32_t>` 按 B32 语义消费 64 lane。即使 Scalar 循环把 `nLanes` 限制为 64，mask 的编码/解释仍不因该整数自动变成 B32；同一 `MaskReg` 跨 B16/B32 使用会破坏 value/index 的逐 lane 对应。 修复把 half/bfloat16/float 的计算类型统一为 float，int32 保持 int32，使计算值 `CT` 与索引都为 4B，二者的 VL 同为 64。B16 输入通过 `DIST_UNPACK_B16` load 后 Reg Cast 到 float，输出在 store 前 Cast 回 B16 并用 `DIST_PACK_B32` 写低 64 个逻辑 lane。比较 mask 与两个 Select 从此共享同一 B32 lane 语义。

## 解决配方

以下 before/after 是修复点两侧完整的 `cummin_scan_vf`，由成功 Edit 事件逐字恢复。

### before

```cpp
__simd_vf__ inline void cummin_scan_vf(
    __ubuf__ T *runCarryVal, __ubuf__ int32_t *runCarryIdx,
    __ubuf__ T *inBuf, __ubuf__ T *outVal, __ubuf__ int32_t *idxI32,
    __ubuf__ int32_t *idxSrc,
    uint32_t nLanes, uint32_t rTile, uint32_t isFirstTile)
{
    constexpr uint32_t VL_T = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    constexpr uint32_t VL_I = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
    AscendC::Reg::RegTensor<T> runReg, xReg;
    AscendC::Reg::RegTensor<int32_t> runIdxReg, curIdxReg;
    AscendC::Reg::MaskReg cmp, maskV;

    // mask: ALWAYS CreateMask<ALL> (full register). The active lanes are the
    // low `nLanes` of the register; lanes [nLanes..VL_T) hold stale / 32B-DMA-
    // pad data. Those high lanes are NEVER output (CopyOut reads only
    // nLanes elements) and vector Compare/Select are lane-independent, so the
    // stale high lanes cannot corrupt the active low-nLanes lanes -- they just
    // compute a don't-care min over garbage that is discarded.
    //
    // This deliberately avoids AscendC::Reg::UpdateMask. Per the MaskReg card
    // UpdateMask is designed for ROLLING consumption (full-mask iterations
    // decrementing a scalar by VL_T, partial mask only as the trailing call
    // AFTER a full iteration). A standalone UpdateMask(nLanes<VL_T) as the
    // sole mask call produces a corrupt mask for Compare/Select on dav-3510
    // (verified by probe: nLanes=8 -> row0 seed correct, row1 first
    // Compare/Select garbage; nLanes=64 full -> clean). CreateMask<ALL> is the
    // verified full-register path (mul.asc, the a5 DoCompare full-iteration
    // path, and our own nLanes=64 cases all pass).
    (void)nLanes;
    maskV = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    if (!isFirstTile) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(runReg, runCarryVal);
        AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(runIdxReg, runCarryIdx);
    }

    for (uint32_t r = 0; r < rTile; ++r) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(xReg, inBuf + r * VL_T);
        AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(curIdxReg, idxSrc + r * VL_I);
        if (isFirstTile && r == 0) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(runReg, inBuf + r * VL_T);
            AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(runIdxReg, idxSrc + r * VL_I);
        } else {
            AscendC::Reg::Compare<T, AscendC::CMPMODE::LT>(cmp, runReg, xReg, maskV);
            AscendC::Reg::Select<T>(runReg, runReg, xReg, cmp);
            AscendC::Reg::Select<int32_t>(runIdxReg, runIdxReg, curIdxReg, cmp);
        }
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            outVal + r * VL_T, runReg, maskV);
        AscendC::Reg::StoreAlign<int32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            idxI32 + r * VL_I, runIdxReg, maskV);
    }

    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        runCarryVal, runReg, maskV);
    AscendC::Reg::StoreAlign<int32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        runCarryIdx, runIdxReg, maskV);
}
```

### after

```cpp
__simd_vf__ inline void cummin_scan_vf(
    __ubuf__ T *runCarryVal, __ubuf__ int32_t *runCarryIdx,
    __ubuf__ T *inBuf, __ubuf__ T *outVal, __ubuf__ int32_t *idxI32,
    __ubuf__ int32_t *idxSrc,
    uint32_t nLanes, uint32_t rTile, uint32_t isFirstTile)
{
    // Compute type: half/bfloat16/float scan in float, int32 scans in int32.
    // Both float and int32 are 4 B/lane so VL_CT = VL_I = 64 -- the value (CT)
    // and index (int32) registers are lane-locked under one b32 MaskReg. For
    // B16 (sizeof(T)==2) VL_T would be 128 != VL_I=64: a b16-typed Compare
    // mask (16 bit, 128-lane) cannot drive a b32 Select<int32> (8 bit,
    // 64-lane), which corrupted half/bf16 argmin indices. The B16 path
    // therefore loads with DIST_UNPACK_B16 (64 B16 elems -> 64-lane register),
    // Reg::Cast<float,T> to a 64-lane float register, scans float+int32, and
    // stores via Reg::Cast<T,float> + DIST_PACK_B32. This is the B16<->float
    // pattern sanctioned by references/reg-api-guide.md. Cast values are
    // bit-exact (every B16 has an exact float32), so the LT ordering and tie
    // (later-index) semantics match torch.cummin on the original B16 input.
    using CT = typename std::conditional<std::is_same<T, int32_t>::value,
                                         int32_t, float>::type;
    constexpr uint32_t VL_T = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    constexpr uint32_t VL_CT = AscendC::VECTOR_REG_WIDTH / sizeof(CT);
    constexpr uint32_t VL_I = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
    static_assert(VL_CT == VL_I, "value and index lanes must be lane-locked");
    static constexpr AscendC::Reg::CastTrait b16ToFloat = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait floatToB16 = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    AscendC::Reg::RegTensor<CT> runReg, xReg;
    AscendC::Reg::RegTensor<T> bReg;                 // B16 scratch (4B path unused)
    AscendC::Reg::RegTensor<int32_t> runIdxReg, curIdxReg;
    AscendC::Reg::MaskReg cmp, maskV;

    // mask: ALWAYS CreateMask<CT,ALL> (full 64-lane b32 register). Active lanes
    // are the low nLanes; high lanes hold stale 32B-DMA-pad data but are never
    // output (CopyOut reads only nLanes) and vector ops are lane-independent,
    // so stale high lanes cannot corrupt the active low-nLanes lanes. This
    // avoids Reg::UpdateMask: per the MaskReg card UpdateMask is for ROLLING
    // consumption (full-mask iters decrement a scalar, partial only as the
    // trailing call AFTER a full iteration); a standalone UpdateMask(nLanes<
    // VL_CT) corrupts Compare/Select on dav-3510 (probe: nLanes=8 -> garbage,
    // nLanes=64 -> clean). CreateMask<ALL> is the verified full-register path.
    (void)nLanes;
    maskV = AscendC::Reg::CreateMask<CT, AscendC::Reg::MaskPattern::ALL>();

    if (!isFirstTile) {
        if constexpr (sizeof(T) == 2) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, runCarryVal);
            AscendC::Reg::Cast<CT, T, b16ToFloat>(runReg, bReg, maskV);
        } else {
            (void)bReg;
            AscendC::Reg::LoadAlign<CT, AscendC::Reg::LoadDist::DIST_NORM>(runReg, runCarryVal);
        }
        AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(runIdxReg, runCarryIdx);
    } else {
        (void)bReg;
    }

    for (uint32_t r = 0; r < rTile; ++r) {
        if constexpr (sizeof(T) == 2) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, inBuf + r * VL_T);
            AscendC::Reg::Cast<CT, T, b16ToFloat>(xReg, bReg, maskV);
        } else {
            AscendC::Reg::LoadAlign<CT, AscendC::Reg::LoadDist::DIST_NORM>(xReg, inBuf + r * VL_T);
        }
        AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(curIdxReg, idxSrc + r * VL_I);
        if (isFirstTile && r == 0) {
            // seed: running = first row. Re-load (xReg already holds row 0;
            // the duplicate UB read is MTE2-cached and negligible).
            if constexpr (sizeof(T) == 2) {
                AscendC::Reg::Cast<CT, T, b16ToFloat>(runReg, bReg, maskV);
            } else {
                AscendC::Reg::LoadAlign<CT, AscendC::Reg::LoadDist::DIST_NORM>(runReg, inBuf + r * VL_T);
            }
            AscendC::Reg::LoadAlign<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(runIdxReg, idxSrc + r * VL_I);
        } else {
            AscendC::Reg::Compare<CT, AscendC::CMPMODE::LT>(cmp, runReg, xReg, maskV);
            AscendC::Reg::Select<CT>(runReg, runReg, xReg, cmp);
            AscendC::Reg::Select<int32_t>(runIdxReg, runIdxReg, curIdxReg, cmp);
        }
        if constexpr (sizeof(T) == 2) {
            AscendC::Reg::Cast<T, CT, floatToB16>(bReg, runReg, maskV);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                outVal + r * VL_T, bReg, maskV);
        } else {
            AscendC::Reg::StoreAlign<CT, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                outVal + r * VL_T, runReg, maskV);
        }
        AscendC::Reg::StoreAlign<int32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            idxI32 + r * VL_I, runIdxReg, maskV);
    }

    if constexpr (sizeof(T) == 2) {
        AscendC::Reg::Cast<T, CT, floatToB16>(bReg, runReg, maskV);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
            runCarryVal, bReg, maskV);
    } else {
        AscendC::Reg::StoreAlign<CT, AscendC::Reg::StoreDist::DIST_NORM_B32>(
            runCarryVal, runReg, maskV);
    }
    AscendC::Reg::StoreAlign<int32_t, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        runCarryIdx, runIdxReg, maskV);
}
```

## 设计决策

选择把 B16 值升到 float，而不是尝试复用 B16 mask，是为了让 value 和 int32 index 的寄存器宽度、MaskReg 解释与 Select lane 一次性对齐。放弃仅把 `nLanes` 设为 64，因为前一轮已经证明 Scalar 活动长度正确仍不足以改变 B16 mask 的类型语义；放弃为索引单独生成另一个比较，因为两份比较可能在 NaN/相等语义上漂移，破坏值与 argmin 同步。

当前方案是 `safety_fallback`。如果目标要求 B16 原生快路径，应设计两个 64-lane 子批或显式 mask 转换，并分别验证 value 与 index 的逐 lane 一致性。

## 预防

- 审查 MaskReg 的生产类型和每个消费者类型，不能只审查 `nLanes` 数值。
- 对 value + index 双输出算子，增加“values 正确但 indices 错”的独立断言，避免值误差掩盖 lane-lock 问题。
- 在 B16、B32 混合链中画出每个 RegTensor 的字节宽度和 VL，再决定 mask 的唯一生产类型。
- Cast 的 load/store distribution 必须与低 64 个逻辑 lane 的布局配套验证。

## 验证信号

验证来自本次 cummin 开发中的远端 Ascend950 NPU pytest，证据等级均为 `tool_observed`。

- 修复前，完成 CopyIn 修正后测试为 94 passed、14 failed；失败项集中在 float16/bfloat16 路径，索引不匹配仍存在（950，tool_observed）。
- 引入 `CT=float`、`DIST_UNPACK_B16`、Reg Cast 与 `DIST_PACK_B32` 后，未改测试容差或参考实现，完整 pytest 达到 108 passed、20 skipped（950，tool_observed）。
- 后续加入 mixed-NaN 回归并完成独立 NaN 修复后，同一最终实现达到 109 passed、20 skipped，最终评测 20/20 正确（950，tool_observed）。

## 失效边界

- 未验证其他 NPU 架构、其他 Reg 宽度或 CANN 版本；不得把 B16/B32 mask 布局假定为跨平台恒定。
- 若伴随 metadata 不是 int32，公共计算类型和 VL 必须重新计算。
- 升到 float 可能改变某些算子的舍入/溢出语义；本卡只覆盖“比较并选择原值、随后回存”的前缀最小路径。
- 本轨迹没有给出该 Cast 路线相对 B16 原生实现的性能结论。

## 坏实践

- 认为 `nLanes=64` 足以让 B16 mask 自动变成 64-lane B32 mask。
- 用 B16 `Compare` 的 mask 直接驱动 int32 `Select`，只检查 values 不检查 indices。
- 为值和索引各算一套独立条件却不验证 NaN、相等与 tie-breaking 一致。

## 误诊

在此之前，部分 B16 失败曾与 CopyIn 尾块错误混在一起。逐行 CopyIn 后 float32/int32 已全过、仅 B16 路径残留，才把根因收敛到跨 dtype mask lane-lock。