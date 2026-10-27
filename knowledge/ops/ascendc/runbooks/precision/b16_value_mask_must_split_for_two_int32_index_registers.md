---
type: "Runbook"
title: "半精度索引错位—拆分双 int32 掩码"
description: "16 位 value 一寄存器对应两组 32 位 index，复用单掩码会漏写半区；用 UnPack 拆高低掩码并双寄存器更新。"
tags: ["task.debugging", "topic.precision", "topic.implementation.masking", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 半精度索引错位—拆分双 int32 掩码

## 现象

在 RegBase 前缀扫描中，value 使用 fp16/bf16，而 argmin 工作区使用 int32。32 位 value 路径能通过；16 位 value 路径在沿外轴扫描时出现 value/index 不一致。使用 `CreateMask<int32_t>` 的单寄存器规避法仍失败，而构造定向输入后可以看到 index 的后一半出现成片错误。

这个问题只在“一个 16 位 value 寄存器需要维护同数量的 32 位 index”时出现，不是比较谓词或容差问题。

## 适用场景

- 工程形态：Ascend C kernel 直调精度与性能联调；
- API 路径：RegBase `RegTensor` / `MaskReg` / `UnPack<LOWEST|HIGHEST>` / `Select` / `StoreAlign`；
- 验证过的平台与 CANN 版本：950 / CANN 9.0.1；
- 适用粒度：`operator_family`，适用于 16 位数据与 32 位伴随元数据按 lane 锁步更新的 vector kernel；
- 路线优先级：`recommended_fastpath`，保留 16 位 RegBase，一次 value 寄存器配两组 int32 index；
- 明确不适用 / 未验证场景：value 与伴随元数据等宽的路径、没有 lane-wise 元数据的算子，以及 A2/A3 未验证。

## 根因

一个 256B 的 fp16/bf16 寄存器有 128 个 value lane，而一个 int32 寄存器只有 64 个 index lane，单个 int32 RegTensor 与单个未拆分 MaskReg 无法覆盖同一批 128 个逻辑元素。 换算是：`256B / 2B = 128` 个 16 位 value，`256B / 4B = 64` 个 int32 index；因此每个 value register iteration 必须加载两组 int32 寄存器，并把 128-bit value mask 分成低 64-bit 与高 64-bit 两个 index mask。

## 解决配方

保持 value 侧按 `vRegSize / sizeof(X)` 计算迭代次数。对 32 位 X 使用单寄存器；对 16 位 X：

1. 创建 X 宽度的 `yMaskReg`；
2. 用 `UnPack<LOWEST/HIGHEST>` 得到两组 int32 store mask；
3. 加载 `argminReg1` 和偏移 `halfRegElems` 的 `argminReg2`；
4. 每轮比较后再次拆分 `keepMinReg`，分别 Select/Store 两组 index；
5. Host Tiling 的 `regComputeTimes/reservedRegTimes` 恢复按 `elemsPerReg=vRegSize/dSize` 计算。

### before

证据定位：revision=轨迹工具写入前状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`DoCompareVF`；行段=完整函数；支撑结论=16 位 value 与 32 位 index 共用单个 64-lane mask/register 路径。

```cpp
template <typename X, bool IS_FLOAT>
__simd_vf__ inline void DoCompareVF(
    __ubuf__ X* selectXAddr,
    __ubuf__ int32_t* selectArgminAddr,
    __ubuf__ X* yAddr,
    __ubuf__ int32_t* argminAddr,
    int64_t computeTimes,
    int64_t regComputeTimes,
    uint32_t reservedRegComputeTimes,
    int64_t allocLength,
    int32_t duplicate,
    uint32_t vRegSize,
    int32_t dSize)
{
    // For dSize==2 (fp16/bf16): we process regWorkElems (int32 width) elements
    // per iteration. The mask is created with int32_t type so it has exactly
    // regWorkElems bits. This mask works correctly for BOTH X<half> ops (covers
    // the first regWorkElems half elements) and int32 ops (covers all
    // regWorkElems int32 elements). No UnPack needed.
    // tmpOffset advances by regWorkElems for BOTH X and int32 pointers.
    int32_t regWorkElems = static_cast<int32_t>(vRegSize / sizeof(int32_t));

    AscendC::Reg::MaskReg yMaskReg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg keepMinReg;
    AscendC::Reg::MaskReg ltReg, nanReg, takeCurrReg;
    AscendC::Reg::RegTensor<X> xReg, yReg;
    AscendC::Reg::RegTensor<int32_t> argminReg, dupReg;

    for (int64_t i = 0; i < regComputeTimes; i++) {
        if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
            uint32_t cnt = reservedRegComputeTimes;
            yMaskReg = AscendC::Reg::UpdateMask<int32_t>(cnt);
        }
        // Advance by regWorkElems (64) in element units for BOTH pointer types.
        // For X* this is 64*dSize bytes; for int32_t* this is 64*4 bytes.
        // Both point to the same logical elements [i*64, i*64+63].
        int64_t tmpOffset = static_cast<int64_t>(static_cast<int32_t>(i) * regWorkElems);
        AscendC::Reg::LoadAlign<X>(xReg, selectXAddr + tmpOffset);
        AscendC::Reg::LoadAlign<int32_t>(argminReg, selectArgminAddr + tmpOffset);
        for (int64_t j = 0; j < computeTimes; j++) {
            int64_t yOffset = tmpOffset + j * allocLength;
            AscendC::Reg::LoadAlign<X>(yReg, yAddr + yOffset);
            int32_t dupVal = static_cast<int32_t>(j + duplicate);
            AscendC::Reg::Duplicate(dupReg, dupVal);
            if constexpr (IS_FLOAT) {
                // PyTorch cummin: last-occurrence for ties → curr <= acc
                AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                AscendC::Reg::Compare<X, AscendC::CMPMODE::NE>(nanReg, yReg, yReg, yMaskReg);
                AscendC::Reg::Or(takeCurrReg, takeCurrReg, nanReg, yMaskReg);
                AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
            } else {
                AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
            }
            AscendC::Reg::Select<X>(xReg, xReg, yReg, keepMinReg);
            AscendC::Reg::Select<int32_t>(argminReg, argminReg, dupReg, keepMinReg);
            AscendC::Reg::StoreAlign<X>(yAddr + yOffset, xReg, yMaskReg);
            AscendC::Reg::StoreAlign<int32_t>(argminAddr + yOffset, argminReg, yMaskReg);
        }
        AscendC::Reg::StoreAlign<X>(selectXAddr + tmpOffset, xReg, yMaskReg);
        AscendC::Reg::StoreAlign<int32_t>(selectArgminAddr + tmpOffset, argminReg, yMaskReg);
    }
}
```

证据定位：revision=轨迹工具写入前状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`calc_cummin_tiling`；行段=完整函数；支撑结论=Host 侧按 int32 lane 宽计算 RegBase 迭代次数。

```cpp
CumminTilingData calc_cummin_tiling(int64_t M, int64_t R, int64_t N,
                                    int32_t dSize, int32_t needCast)
{
    CumminTilingData t;
    t.M = M; t.R = R; t.N = N;
    t.dSize = dSize; t.needCast = needCast;
    t.vRegSize = 256;
    t.elemsPerReg = t.vRegSize / dSize;  // for buffer padding (LoadAlign reads full reg)

    // For dSize==2 (fp16/bf16), DoCompareVF processes int32-width (64) elements
    // per register iteration to keep X and int32 masks compatible. For dSize==4,
    // X and int32 have the same width (64), so no change.
    int32_t regWorkElems = t.vRegSize / sizeof(int32_t);  // always 64

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int32_t argminDSize = sizeof(int64_t);
    int64_t alignElems = 32 / dSize;
    int64_t alignedN = HOST_CEIL_DIV(N, alignElems) * alignElems;
    if (alignedN == 0) alignedN = alignElems;
    t.alignedN = alignedN;

    int64_t ubAvail = static_cast<int64_t>(ubSize) - RESERVED_UB;
    // Buffer layout: y (dSize per elem) + argmin int32 (4 per elem) + cast int64 (8 per elem)
    // Use conservative formula: y gets 1/3 of UB, argmin+cast get 2/3
    int64_t ubSize_ = ubAvail / (1 + static_cast<int64_t>(sizeof(int32_t)) / dSize + static_cast<int64_t>(sizeof(int64_t)) / dSize);

    t.baseRows = M / coreNum;
    t.extraRows = M % coreNum;

    // SIMT coverage (priority) — only when N is tiny, M is large, R is small
    // (vector SIMD waste + enough threads for parallelism). fp16/bf16 now use
    // RegBase with UnPack dual-mask (golden cummin.h:459-507), not forced SIMT.
    if (N < SIMT_N_THRESHOLD && M > SIMT_M_THRESHOLD && R < SIMT_R_THRESHOLD) {
        t.tilingKey = CUMMIN_KEY_COMPUTE_SIMT;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        return t;
    }

    // key0: RN Full Load
    // Buffer: y(dSize) + argmin_int32(4) + argmin_int64(8) per element + fixed
    int64_t regElems0 = t.elemsPerReg;
    int64_t dupCount0 = HOST_CEIL_DIV(alignedN, 8) * 8;
    int64_t bytesPerM0 = R * alignedN * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)));
    int64_t fixedM0 = regElems0 * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)))
                    + (alignedN + regElems0) * dSize
                    + (dupCount0 + regElems0) * static_cast<int64_t>(sizeof(int32_t));
    int64_t computeM = (ubAvail - fixedM0) / bytesPerM0;

    if (computeM >= 1) {
        t.tilingKey = CUMMIN_KEY_RN_FULLY_LOAD;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        t.computeM = computeM;
        int64_t mPerCore = t.baseRows + (t.extraRows > 0 ? 1 : 0);
        t.splitM = HOST_CEIL_DIV(mPerCore, computeM);
        t.reservedM = mPerCore - (t.splitM - 1) * computeM;
        if (t.reservedM <= 0) { t.reservedM = computeM; t.splitM -= 1; }
        if (t.reservedM == computeM) t.reservedM = 0;
        // precompute reg-level values (workWidth = alignedN)
        t.regComputeTimes = HOST_CEIL_DIV(alignedN, regWorkElems);
        t.reservedRegTimes = alignedN % regWorkElems;
        if (t.reservedRegTimes == 0) t.reservedRegTimes = regWorkElems;
        if (t.reservedRegTimes == regWorkElems) t.reservedRegTimes = 0;
        t.dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
        t.dataCopyBlockLen = alignedN * dSize / 32;
        return t;
    }

    // key1/key2: account for 3 buffers (y + argmin int32 + cast int64) + padding + fixed buffers
    int64_t regElems = t.elemsPerReg;
    int64_t dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
    int64_t bytesPerLine = alignedN * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)));
    // Fixed: y padding + argmin padding + xTmp + argminTmp
    int64_t fixedBytes = regElems * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)))
                       + (alignedN + regElems) * dSize
                       + (dupCount + regElems) * static_cast<int64_t>(sizeof(int32_t));
    int64_t ubLines = (ubAvail - fixedBytes) / bytesPerLine;

    if (ubLines >= 2) {
        t.tilingKey = CUMMIN_KEY_N_FULLY_LOAD;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        t.computeR = ubLines;
        t.splitR = HOST_CEIL_DIV(R, t.computeR);
        t.reservedR = R - (t.splitR - 1) * t.computeR;
        if (t.reservedR == t.computeR) t.reservedR = 0;
        // precompute reg-level values (workWidth = alignedN)
        t.regComputeTimes = HOST_CEIL_DIV(alignedN, regWorkElems);
        t.reservedRegTimes = alignedN % regWorkElems;
        if (t.reservedRegTimes == 0) t.reservedRegTimes = regWorkElems;
        if (t.reservedRegTimes == regWorkElems) t.reservedRegTimes = 0;
        t.dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
        t.dataCopyBlockLen = alignedN * dSize / 32;
        return t;
    }

    // key2: N batched
    t.tilingKey = CUMMIN_KEY_N_NOT_FULLY_LOAD;
    t.blockDim = HOST_MIN(M, coreNum);
    t.baseRows = M / t.blockDim;
    t.extraRows = M % t.blockDim;
    // key2: N batched - account for 3 buffers (y + argmin + cast)
    int64_t bytesPerN = dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t));
    int64_t vRegSize = 256;
    int64_t computeN = (ubAvail / 3 / bytesPerN);  // divide by 3 for y+argmin+cast
    computeN = HOST_CEIL_DIV(computeN, vRegSize / dSize) * (vRegSize / dSize);
    if (computeN < alignElems) computeN = alignElems;
    t.computeN = computeN;
    t.splitN = HOST_CEIL_DIV(N, computeN);
    t.reservedN = N - (t.splitN - 1) * computeN;
    if (t.reservedN == computeN) t.reservedN = 0;
    // precompute reg-level values (workWidth = computeN)
    t.regComputeTimes = HOST_CEIL_DIV(computeN, regWorkElems);
    t.reservedRegTimes = computeN % regWorkElems;
    if (t.reservedRegTimes == 0) t.reservedRegTimes = regWorkElems;
    if (t.reservedRegTimes == regWorkElems) t.reservedRegTimes = 0;
    t.dupCount = HOST_CEIL_DIV(computeN, 8) * 8;
    t.dataCopyBlockLen = computeN * dSize / 32;
    return t;
}
```

### after

证据定位：revision=轨迹工具写入后状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`DoCompareVF`；行段=完整函数；支撑结论=16 位路径把 value mask 拆为两个 int32 mask，并使用两组 index register。

```cpp
template <typename X, bool IS_FLOAT>
__simd_vf__ inline void DoCompareVF(
    __ubuf__ X* selectXAddr,
    __ubuf__ int32_t* selectArgminAddr,
    __ubuf__ X* yAddr,
    __ubuf__ int32_t* argminAddr,
    int64_t computeTimes,
    int64_t regComputeTimes,
    uint32_t reservedRegComputeTimes,
    int64_t allocLength,
    int32_t duplicate,
    uint32_t vRegSize,
    int32_t dSize)
{
    // Golden reference (cummin.h:419-509): for sizeof(X)==4, single mask/register.
    // For sizeof(X)==2, X has 128 elems/reg but int32 has 64 elems/reg.
    // UnPack splits the 128-bit X mask into two 64-bit int32 masks.
    // tmpOffset = i * (vRegSize / dSize) in X-element units.
    if constexpr (sizeof(X) == 4) {
        // fp32/int32: single mask, single argminReg
        AscendC::Reg::MaskReg yMaskReg = AscendC::Reg::CreateMask<X, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg keepMinReg;
        AscendC::Reg::MaskReg ltReg, nanReg, takeCurrReg;
        AscendC::Reg::RegTensor<X> xReg, yReg;
        AscendC::Reg::RegTensor<int32_t> argminReg, dupReg;

        for (int64_t i = 0; i < regComputeTimes; i++) {
            if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
                uint32_t cnt = reservedRegComputeTimes;
                yMaskReg = AscendC::Reg::UpdateMask<X>(cnt);
            }
            int64_t tmpOffset = static_cast<int64_t>(static_cast<int32_t>(i) * static_cast<int32_t>(vRegSize / dSize));
            AscendC::Reg::LoadAlign<X>(xReg, selectXAddr + tmpOffset);
            AscendC::Reg::LoadAlign<int32_t>(argminReg, selectArgminAddr + tmpOffset);
            for (int64_t j = 0; j < computeTimes; j++) {
                int64_t yOffset = tmpOffset + j * allocLength;
                AscendC::Reg::LoadAlign<X>(yReg, yAddr + yOffset);
                int32_t dupVal = static_cast<int32_t>(j + duplicate);
                AscendC::Reg::Duplicate(dupReg, dupVal);
                if constexpr (IS_FLOAT) {
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::NE>(nanReg, yReg, yReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, takeCurrReg, nanReg, yMaskReg);
                    AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
                } else {
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                    AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
                }
                AscendC::Reg::Select<X>(xReg, xReg, yReg, keepMinReg);
                AscendC::Reg::Select<int32_t>(argminReg, argminReg, dupReg, keepMinReg);
                AscendC::Reg::StoreAlign<X>(yAddr + yOffset, xReg, yMaskReg);
                AscendC::Reg::StoreAlign<int32_t>(argminAddr + yOffset, argminReg, yMaskReg);
            }
            AscendC::Reg::StoreAlign<X>(selectXAddr + tmpOffset, xReg, yMaskReg);
            AscendC::Reg::StoreAlign<int32_t>(selectArgminAddr + tmpOffset, argminReg, yMaskReg);
        }
    } else {
        // fp16/bf16: X has 128 elems/reg, int32 has 64. UnPack splits mask.
        AscendC::Reg::MaskReg yMaskReg = AscendC::Reg::CreateMask<X, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg argminMaskReg1, argminMaskReg2;
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(argminMaskReg1, yMaskReg);
        AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(argminMaskReg2, yMaskReg);

        AscendC::Reg::MaskReg keepMinReg;
        AscendC::Reg::MaskReg ltReg, nanReg, takeCurrReg;
        AscendC::Reg::MaskReg cmpReg1, cmpReg2;
        AscendC::Reg::RegTensor<X> xReg, yReg;
        AscendC::Reg::RegTensor<int32_t> argminReg1, argminReg2, dupReg;

        int32_t halfRegElems = static_cast<int32_t>(vRegSize / sizeof(int32_t));  // 64

        for (int64_t i = 0; i < regComputeTimes; i++) {
            if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
                uint32_t cnt = reservedRegComputeTimes;
                yMaskReg = AscendC::Reg::UpdateMask<X>(cnt);
                AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(argminMaskReg1, yMaskReg);
                AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(argminMaskReg2, yMaskReg);
            }
            int64_t tmpOffset = static_cast<int64_t>(static_cast<int32_t>(i) * static_cast<int32_t>(vRegSize / dSize));
            AscendC::Reg::LoadAlign<X>(xReg, selectXAddr + tmpOffset);
            AscendC::Reg::LoadAlign<int32_t>(argminReg1, selectArgminAddr + tmpOffset);
            AscendC::Reg::LoadAlign<int32_t>(argminReg2, selectArgminAddr + tmpOffset + halfRegElems);
            for (int64_t j = 0; j < computeTimes; j++) {
                int64_t yOffset = tmpOffset + j * allocLength;
                AscendC::Reg::LoadAlign<X>(yReg, yAddr + yOffset);
                int32_t dupVal = static_cast<int32_t>(j + duplicate);
                AscendC::Reg::Duplicate(dupReg, dupVal);
                if constexpr (IS_FLOAT) {
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::NE>(nanReg, yReg, yReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, takeCurrReg, nanReg, yMaskReg);
                    AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
                } else {
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::LT>(ltReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Compare<X, AscendC::CMPMODE::EQ>(keepMinReg, yReg, xReg, yMaskReg);
                    AscendC::Reg::Or(takeCurrReg, ltReg, keepMinReg, yMaskReg);
                    AscendC::Reg::Not(keepMinReg, takeCurrReg, yMaskReg);
                }
                AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(cmpReg1, keepMinReg);
                AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(cmpReg2, keepMinReg);
                AscendC::Reg::Select<X>(xReg, xReg, yReg, keepMinReg);
                AscendC::Reg::Select<int32_t>(argminReg1, argminReg1, dupReg, cmpReg1);
                AscendC::Reg::Select<int32_t>(argminReg2, argminReg2, dupReg, cmpReg2);
                AscendC::Reg::StoreAlign<X>(yAddr + yOffset, xReg, yMaskReg);
                AscendC::Reg::StoreAlign<int32_t>(argminAddr + yOffset, argminReg1, argminMaskReg1);
                AscendC::Reg::StoreAlign<int32_t>(argminAddr + yOffset + halfRegElems, argminReg2, argminMaskReg2);
            }
            AscendC::Reg::StoreAlign<X>(selectXAddr + tmpOffset, xReg, yMaskReg);
            AscendC::Reg::StoreAlign<int32_t>(selectArgminAddr + tmpOffset, argminReg1, argminMaskReg1);
            AscendC::Reg::StoreAlign<int32_t>(selectArgminAddr + tmpOffset + halfRegElems, argminReg2, argminMaskReg2);
        }
    }
}
```

证据定位：revision=轨迹工具写入后状态（未绑定 Git revision）；path=`example/csrc/ops/cummin/op_kernel/cummin_kernel.cpp`；symbol=`calc_cummin_tiling`；行段=完整函数；支撑结论=Host 侧恢复按 value dtype 的寄存器 lane 数计算迭代与尾块。

```cpp
CumminTilingData calc_cummin_tiling(int64_t M, int64_t R, int64_t N,
                                    int32_t dSize, int32_t needCast)
{
    CumminTilingData t;
    t.M = M; t.R = R; t.N = N;
    t.dSize = dSize; t.needCast = needCast;
    t.vRegSize = 256;
    // elemsPerReg = vRegSize / dSize (128 for fp16, 64 for fp32). Used for
    // regComputeTimes (matching golden cummin.h: VREGTENSOR_SIZE / dSize).
    // For fp16, DoCompareVF uses UnPack to split 128-bit X mask into two
    // 64-bit int32 masks for dual argmin registers.
    t.elemsPerReg = t.vRegSize / dSize;  // X elements per register (128 for fp16, 64 for fp32)

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int32_t argminDSize = sizeof(int64_t);
    int64_t alignElems = 32 / dSize;
    int64_t alignedN = HOST_CEIL_DIV(N, alignElems) * alignElems;
    if (alignedN == 0) alignedN = alignElems;
    t.alignedN = alignedN;

    int64_t ubAvail = static_cast<int64_t>(ubSize) - RESERVED_UB;
    // Buffer layout: y (dSize per elem) + argmin int32 (4 per elem) + cast int64 (8 per elem)
    // Use conservative formula: y gets 1/3 of UB, argmin+cast get 2/3
    int64_t ubSize_ = ubAvail / (1 + static_cast<int64_t>(sizeof(int32_t)) / dSize + static_cast<int64_t>(sizeof(int64_t)) / dSize);

    t.baseRows = M / coreNum;
    t.extraRows = M % coreNum;

    // SIMT coverage (priority) — only when N is tiny, M is large, R is small
    // (vector SIMD waste + enough threads for parallelism). fp16/bf16 now use
    // RegBase with UnPack dual-mask (golden cummin.h:459-507), not forced SIMT.
    if (N < SIMT_N_THRESHOLD && M > SIMT_M_THRESHOLD && R < SIMT_R_THRESHOLD) {
        t.tilingKey = CUMMIN_KEY_COMPUTE_SIMT;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        return t;
    }

    // key0: RN Full Load
    // Buffer: y(dSize) + argmin_int32(4) + argmin_int64(8) per element + fixed
    int64_t regElems0 = t.elemsPerReg;
    int64_t dupCount0 = HOST_CEIL_DIV(alignedN, 8) * 8;
    int64_t bytesPerM0 = R * alignedN * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)));
    int64_t fixedM0 = regElems0 * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)))
                    + (alignedN + regElems0) * dSize
                    + (dupCount0 + regElems0) * static_cast<int64_t>(sizeof(int32_t));
    int64_t computeM = (ubAvail - fixedM0) / bytesPerM0;

    if (computeM >= 1) {
        t.tilingKey = CUMMIN_KEY_RN_FULLY_LOAD;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        t.computeM = computeM;
        int64_t mPerCore = t.baseRows + (t.extraRows > 0 ? 1 : 0);
        t.splitM = HOST_CEIL_DIV(mPerCore, computeM);
        t.reservedM = mPerCore - (t.splitM - 1) * computeM;
        if (t.reservedM <= 0) { t.reservedM = computeM; t.splitM -= 1; }
        if (t.reservedM == computeM) t.reservedM = 0;
        // precompute reg-level values (workWidth = alignedN)
        t.regComputeTimes = HOST_CEIL_DIV(alignedN, t.elemsPerReg);
        t.reservedRegTimes = alignedN % t.elemsPerReg;
        if (t.reservedRegTimes == 0) t.reservedRegTimes = t.elemsPerReg;
        if (t.reservedRegTimes == t.elemsPerReg) t.reservedRegTimes = 0;
        t.dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
        t.dataCopyBlockLen = alignedN * dSize / 32;
        return t;
    }

    // key1/key2: account for 3 buffers (y + argmin int32 + cast int64) + padding + fixed buffers
    int64_t regElems = t.elemsPerReg;
    int64_t dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
    int64_t bytesPerLine = alignedN * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)));
    // Fixed: y padding + argmin padding + xTmp + argminTmp
    int64_t fixedBytes = regElems * (dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t)))
                       + (alignedN + regElems) * dSize
                       + (dupCount + regElems) * static_cast<int64_t>(sizeof(int32_t));
    int64_t ubLines = (ubAvail - fixedBytes) / bytesPerLine;

    if (ubLines >= 2) {
        t.tilingKey = CUMMIN_KEY_N_FULLY_LOAD;
        t.blockDim = HOST_MIN(M, coreNum);
        t.baseRows = M / t.blockDim;
        t.extraRows = M % t.blockDim;
        t.computeR = ubLines;
        t.splitR = HOST_CEIL_DIV(R, t.computeR);
        t.reservedR = R - (t.splitR - 1) * t.computeR;
        if (t.reservedR == t.computeR) t.reservedR = 0;
        // precompute reg-level values (workWidth = alignedN)
        t.regComputeTimes = HOST_CEIL_DIV(alignedN, t.elemsPerReg);
        t.reservedRegTimes = alignedN % t.elemsPerReg;
        if (t.reservedRegTimes == 0) t.reservedRegTimes = t.elemsPerReg;
        if (t.reservedRegTimes == t.elemsPerReg) t.reservedRegTimes = 0;
        t.dupCount = HOST_CEIL_DIV(alignedN, 8) * 8;
        t.dataCopyBlockLen = alignedN * dSize / 32;
        return t;
    }

    // key2: N batched
    t.tilingKey = CUMMIN_KEY_N_NOT_FULLY_LOAD;
    t.blockDim = HOST_MIN(M, coreNum);
    t.baseRows = M / t.blockDim;
    t.extraRows = M % t.blockDim;
    // key2: N batched - account for 3 buffers (y + argmin + cast)
    int64_t bytesPerN = dSize + static_cast<int64_t>(sizeof(int32_t)) + static_cast<int64_t>(sizeof(int64_t));
    int64_t vRegSize = 256;
    int64_t computeN = (ubAvail / 3 / bytesPerN);  // divide by 3 for y+argmin+cast
    computeN = HOST_CEIL_DIV(computeN, vRegSize / dSize) * (vRegSize / dSize);
    if (computeN < alignElems) computeN = alignElems;
    t.computeN = computeN;
    t.splitN = HOST_CEIL_DIV(N, computeN);
    t.reservedN = N - (t.splitN - 1) * computeN;
    if (t.reservedN == computeN) t.reservedN = 0;
    // precompute reg-level values (workWidth = computeN)
    t.regComputeTimes = HOST_CEIL_DIV(computeN, t.elemsPerReg);
    t.reservedRegTimes = computeN % t.elemsPerReg;
    if (t.reservedRegTimes == 0) t.reservedRegTimes = t.elemsPerReg;
    if (t.reservedRegTimes == t.elemsPerReg) t.reservedRegTimes = 0;
    t.dupCount = HOST_CEIL_DIV(computeN, 8) * 8;
    t.dataCopyBlockLen = computeN * dSize / 32;
    return t;
}
```

## 设计决策

- 没有继续用 `CreateMask<int32_t>` 驱动 half 比较：实测它会让 16 位 value 路径也失败；
- 没有把每次迭代收缩到 64 个 16 位元素：那会浪费 value 寄存器并且仍混淆 mask 的类型语义；
- 最终采用一组 128-lane value 寄存器 + 两组 64-lane int32 寄存器，属于 `recommended_fastpath`；
- 临时强制 16 位 dtype 走 SIMT 只用于隔离正确性，未作为本卡方案。

## 预防

- 任何 value 与伴随 metadata 异宽的 RegBase 设计，都先写出 `VREG_BYTES / sizeof(dtype)` 的 lane 对照表；
- mask 不只决定“开多少 bit”，还必须匹配被操作寄存器的 lane 解释；
- tail mask 更新后同步重新 `UnPack` 两个子 mask；
- Host Tiling 与 Kernel 必须对同一“每轮逻辑元素数”达成一致。

## 验证信号

来源是同一次 16 位 RegBase 恢复实验：先用单 int32 mask 复现失败，再改为双 UnPack mask 并恢复 X-width tiling。

- 单 int32 mask 方案下，fp16/bf16 外轴路径的 value 与 index 均失败，而 fp32/int32 路径通过（950，证据等级: tool_observed）；
- 双 UnPack 方案构建成功，fp16、bf16、fp32、int32 在两个轴方向的 value/index 均精确匹配 reference（950，证据等级: tool_observed）；
- 完整 20/20 评测通过，浮点输出 MERE/MARE 为 0，int64 index exact match（950，证据等级: tool_observed）。

## 失效边界

- 本配方依赖 950/CANN 9.0.1 的 `AscendC::Reg::UnPack` 与 256B 向量寄存器行为；
- 伴随元数据不是 int32、向量寄存器字节数变化或 layout 不是连续 lane 时，需重新推导寄存器组数与 offset；
- A2、A3 未实测。

## 坏实践

把 `CreateMask<int32_t>` 产生的单 mask 同时用于 half/bfloat16 比较和 int32 index 更新，或只分割 mask 却仍只加载一组 index 寄存器。

## 误诊

早期看到“values 正确、indices 错”时只调整 offset 或把 16 位路径强制到 SIMT；前者没有解决 lane 数不对称，后者只是绕开 RegBase 机制。