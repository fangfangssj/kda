---
type: "Runbook"
title: "合并尾块的非 32B 行导致 PathB 数据损坏"
description: "PathB 把 inner 余数合入最后一个二维 DataCopyPad 块后，blockLen 不再是 32B 的整数倍并产生值与索引损坏；保留完整 V32 块、将余数逐行走一维拷贝可恢复正确性。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 合并尾块的非 32B 行导致 PathB 数据损坏

## 现象

在 Ascend950/dav-3510、CANN 9.0.1 上验证 Cummin 直调 kernel 时，小 inner 的连续 PathA 正常，大 inner 的分块 PathB 出现稳定精度故障。完整测试为 `58 passed, 9 failed`，失败横跨 fp16、fp32、int32 与 bf16。

定向打印显示同一行开头 lane 正确，而尾部列出现损坏值和非法索引；观测包含 `max_diff=nan`、约 `1.2694e38` 的异常差值、int32 的 `802, 0, 802` 破坏序列，以及 `-4787326402820046848` 这样的无效 int64 索引。这不是 Cummin tie/NaN 语义误差，而是搬运后值与索引缓冲区已被破坏。

## 适用场景

- 工程形态：Ascend C 直调 kernel，把非连续 inner lane 分块后用二维 `DataCopyPad` 跨 axis 行搬运。
- API 路径：二维 `DataCopyPad` 的 `blockLen`、`srcStride`/`dstStride` 与余数块处理。
- 已验证平台与版本：Ascend950/dav-3510，CANN 9.0.1。
- 适用粒度：`operator_family`，适用于同类 strided DMA 扫描或归约 kernel 的 inner 尾块。
- 路线优先级：`safety_fallback`。余数逐行走一维搬运以保证正确性；若需高性能，应在有独立验证的合法对齐方案上再优化。
- 不适用：本来就是连续的一维尾块，或二维搬运的每行长度已满足目标 API 对齐契约时，不应机械改为逐行。

## 根因

原 tiling 先取 `gFull = 32 / sizeof(T)`，再用 floor 计算 chunk 数，并把余数 `r` 合入最后块： ```text gLast = gFull + r blockLen = gLast * sizeof(T) = 32 + r * sizeof(T) ``` 只要 `r != 0`，最后一个二维块的每行长度就不是 32B 的整数倍。例如元素宽 4B 且余数为 1 时，最后行长为 36B。`Process` 仍把这个块判定为非连续 PathB，并以二维 `DataCopyPad` 跨多行搬运。失败仅集中在带余数的大-inner 路径，定向打印又把损坏定位到尾部列；保留完整 32B 行、把余数改成每次一行的一维搬运后全部测试通过，形成单根因闭环。

## 解决配方

代码来自未提交开发态 `cummin.asc`。before/after 完整给出负责 chunk 解释的 `KernelCummin<T>::Init`、搬运路线选择的 `KernelCummin<T>::Process`，以及生成 chunk 元数据的 `calc_cummin_tiling`。

### before

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                               int64_t outer, int64_t axis, int64_t inner,
                               int64_t totalLanes, int64_t blockLanes,
                               int64_t nChunks, int64_t gFull, int64_t gLast)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        vGm_.SetGlobalBuffer((__gm__ T *)values);
        iGm_.SetGlobalBuffer((__gm__ int64_t *)indices);
        outer_ = outer; axis_ = axis; inner_ = inner;
        totalLanes_ = totalLanes; blockLanes_ = blockLanes;
        nChunks_ = nChunks; gFull_ = gFull; gLast_ = gLast;
        // UB buffers sized to the max tile (TILE elements).
        pipe_.InitBuffer(inQ_,  PIPELINE_DEPTH, TILE * sizeof(T));
        pipe_.InitBuffer(outV_, PIPELINE_DEPTH, TILE * sizeof(T));
        pipe_.InitBuffer(outI_, PIPELINE_DEPTH, TILE * sizeof(int64_t));
    }
```

```cpp
    __aicore__ inline void Process()
    {
        int64_t laneLo = blockLanes_ * AscendC::GetBlockIdx();
        int64_t hi = laneLo + blockLanes_;
        if (hi > totalLanes_) hi = totalLanes_;

        for (int64_t L = laneLo; L < hi; ++L) {
            int64_t o, k;
            if (nChunks_ > 1) { k = L % nChunks_; o = L / nChunks_; }
            else { k = 0; o = L; }
            int64_t lanes = (k == nChunks_ - 1) ? gLast_ : gFull_;
            int64_t i0 = k * gFull_;                 // start inner offset of this chunk
            ResetCarry(lanes);

            int64_t A = TILE / lanes;                // rows per tile
            if (A < 1) A = 1;
            int64_t aBase = 0;
            while (aBase < axis_) {
                int64_t cntA = A;
                if (aBase + cntA > axis_) cntA = axis_ - aBase;   // tail axisChunk
                int64_t base = o * axis_ * inner_ + aBase * inner_ + i0;
                bool contig = (lanes == inner_);     // PathA (incl inner==1)
                CopyIn(base, cntA, lanes, contig);
                Scan(cntA, lanes, aBase);
                CopyOut(base, cntA, lanes, contig);
                aBase += cntA;
            }
        }
    }
```

```cpp
CumminTiling calc_cummin_tiling(int64_t outer, int64_t inner, int64_t sizeofT)
{
    constexpr static int64_t INNER_CAP = 64;     // PathA threshold
    constexpr static int64_t MIN_LANES_PER_CORE = 1;
    int64_t nChunks, gFull, gLast;
    if (inner <= INNER_CAP) {
        // PathA: one chunk of all `inner` lanes, contiguous 1D load.
        nChunks = 1;
        gFull = inner;
        gLast = inner;
    } else {
        // PathB: chunk by g (=32/sizeof so blockLen=32B); merge tail into last chunk.
        gFull = 32 / sizeofT;
        if (gFull < 1) gFull = 1;
        if (gFull > MAX_LANES / 2) gFull = MAX_LANES / 2;
        nChunks = inner / gFull;                 // floor; >=1 since inner>64>=gFull
        if (nChunks < 1) nChunks = 1;
        gLast = inner - (nChunks - 1) * gFull;   // in [gFull, 2*gFull-1]
    }
    int64_t totalLanes = outer * nChunks;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = totalLanes / MIN_LANES_PER_CORE;
    if (numBlocks > coreNum) numBlocks = coreNum;
    if (numBlocks < 1) numBlocks = 1;
    int64_t blockLanes = (totalLanes + numBlocks - 1) / numBlocks;

    CumminTiling t;
    t.numBlocks = numBlocks; t.blockLanes = blockLanes;
    t.nChunks = nChunks; t.gFull = gFull; t.gLast = gLast; t.totalLanes = totalLanes;
    return t;
}
```

### after

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                               int64_t outer, int64_t axis, int64_t inner,
                               int64_t totalLanes, int64_t blockLanes,
                               int64_t nChunks, int64_t gFull, int64_t tailRem)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)x);
        vGm_.SetGlobalBuffer((__gm__ T *)values);
        iGm_.SetGlobalBuffer((__gm__ int64_t *)indices);
        outer_ = outer; axis_ = axis; inner_ = inner;
        totalLanes_ = totalLanes; blockLanes_ = blockLanes;
        nChunks_ = nChunks; gFull_ = gFull; tailRem_ = tailRem;
        nFull_ = (tailRem_ > 0) ? (nChunks_ - 1) : nChunks_;
        // UB buffers sized to the max tile (TILE elements).
        pipe_.InitBuffer(inQ_,  PIPELINE_DEPTH, TILE * sizeof(T));
        pipe_.InitBuffer(outV_, PIPELINE_DEPTH, TILE * sizeof(T));
        pipe_.InitBuffer(outI_, PIPELINE_DEPTH, TILE * sizeof(int64_t));
    }
```

```cpp
    __aicore__ inline void Process()
    {
        int64_t laneLo = blockLanes_ * AscendC::GetBlockIdx();
        int64_t hi = laneLo + blockLanes_;
        if (hi > totalLanes_) hi = totalLanes_;

        for (int64_t L = laneLo; L < hi; ++L) {
            int64_t o, k;
            if (nChunks_ > 1) { k = L % nChunks_; o = L / nChunks_; }
            else { k = 0; o = L; }
            // tail chunk (only when tailRem_>0) is the last inner-chunk.
            bool isTail = (tailRem_ > 0 && k == nFull_);
            int64_t lanes = isTail ? tailRem_ : gFull_;
            int64_t i0 = k * gFull_;                 // start inner offset of this chunk
            ResetCarry(lanes);

            // PathA / tail: 1D per-tile (tail uses A=1, one row at a time since rows
            // are strided). PathB full: 2D strided, A = TILE/lanes.
            int64_t A;
            bool contig;
            if (isTail) { A = 1; contig = true; }             // 1D, row-by-row
            else if (lanes == inner_) { A = TILE / lanes; contig = true; }  // PathA contiguous
            else { A = TILE / lanes; contig = false; }        // PathB 2D
            if (A < 1) A = 1;

            int64_t aBase = 0;
            while (aBase < axis_) {
                int64_t cntA = A;
                if (aBase + cntA > axis_) cntA = axis_ - aBase;   // tail axisChunk
                int64_t base = o * axis_ * inner_ + aBase * inner_ + i0;
                CopyIn(base, cntA, lanes, contig);
                Scan(cntA, lanes, aBase);
                CopyOut(base, cntA, lanes, contig);
                aBase += cntA;
            }
        }
    }
```

```cpp
CumminTiling calc_cummin_tiling(int64_t outer, int64_t inner, int64_t sizeofT)
{
    constexpr static int64_t INNER_CAP = 64;     // PathA threshold
    constexpr static int64_t MIN_LANES_PER_CORE = 1;
    int64_t nChunks, gFull, tailRem;
    if (inner <= INNER_CAP) {
        // PathA: one chunk of all `inner` lanes, contiguous 1D load.
        nChunks = 1;
        gFull = inner;
        tailRem = 0;
    } else {
        // PathB: chunk by V32 (=32/sizeof so full-chunk blockLen = 32B, 32B-multiple).
        // Remainder handled as a separate 1D per-row chunk (rows are contiguous).
        gFull = 32 / sizeofT;
        if (gFull < 1) gFull = 1;
        if (gFull > MAX_LANES / 2) gFull = MAX_LANES / 2;
        int64_t nFull = inner / gFull;            // >=1 since inner>64>=gFull
        if (nFull < 1) nFull = 1;
        tailRem = inner - nFull * gFull;           // in [0, gFull-1]
        nChunks = nFull + (tailRem > 0 ? 1 : 0);
    }
    int64_t totalLanes = outer * nChunks;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = totalLanes / MIN_LANES_PER_CORE;
    if (numBlocks > coreNum) numBlocks = coreNum;
    if (numBlocks < 1) numBlocks = 1;
    int64_t blockLanes = (totalLanes + numBlocks - 1) / numBlocks;

    CumminTiling t;
    t.numBlocks = numBlocks; t.blockLanes = blockLanes;
    t.nChunks = nChunks; t.gFull = gFull; t.tailRem = tailRem; t.totalLanes = totalLanes;
    return t;
}
```

## 设计决策

- 完整 chunk 固定为 `V32 = 32 / sizeof(T)` 个元素，使二维搬运的每行 `blockLen` 恰为 32B。
- 余数成为独立 chunk；`Process` 以 `A=1`、`contig=true` 一次处理一行，避免把跨行 stride 与不足 32B 的尾行组合为二维搬运。
- `nFull_` 由 `nChunks_` 与 `tailRem_` 推导，只有真实存在余数时最后 chunk 才走 fallback，整除输入仍全部走二维 PathB。
- 该路线按正确性归类为 `safety_fallback`，没有证据证明逐行余数是性能最优实现。

## 预防

- 设计 strided `DataCopyPad` tiling 时，对每个 chunk 显式计算字节级 `blockLen`；不能只验证元素数不越界。
- 将“完整对齐块”和“余数块”建模为不同路线，不要为了减少 chunk 数把余数合入最后一个二维块。
- 回归必须覆盖各 dtype 的大 inner 非整除情况，并同时核对 values 与 indices；只看开头 lane 会漏掉尾部损坏。

## 验证信号

验证来自该 Cummin 开发会话中的同一源码失败、定向诊断、单一修复和最终上板回归。

- 修复前完整 NPU pytest 为 `58 passed, 9 failed`；九个失败均是大-inner PathB，覆盖 fp16、fp32、int32 与 bf16（来源平台: 950/dav-3510；证据等级: tool_observed）。
- 定向打印显示尾部列含损坏数值和非法索引，而早期 lane 正确，排除了全局扫描语义错误（来源平台: 950/dav-3510；证据等级: tool_observed）。
- 切换为完整 V32 二维块加逐行一维余数后，首次全套回归为 `67 passed`；清洁复跑仍为 `67 passed`（来源平台: 950/dav-3510；证据等级: tool_observed）。
- 标准评测 `20/20` 通过，values 在容差内且 indices exact match（来源平台: 950/dav-3510；证据等级: tool_observed）。

## 性能影响

最终正确性优先实现相对 benchmark reference 的整体平均速度约为 `0.03x`。该数字包含 scalar scan、整体 tiling 与搬运策略，轨迹没有隔离逐行 tail fallback 的成本，因此不能把 `0.03x` 归因于本修复；它只说明当前方案适合作为 correctness `safety_fallback`，不构成性能 fast path。

## 失效边界

- 低层 DMA trace 未被记录；根因由失败分布、尾列损坏与单一修复后的全量通过共同支持，而非硬件事务 dump。
- 只验证 Ascend950/dav-3510、CANN 9.0.1；其他平台的二维搬运约束需重新查证并实测。
- 对齐且没有余数的 PathB 不走逐行路线；不要把 fallback 扩大到所有 chunk。
- 本卡不证明一维余数搬运的吞吐最优，也不提供其他合法二维尾块编码。

## 坏实践

- 用 floor chunk 数后把余数并入最后一块，却仍假设所有二维行都是 32B。
- 为让测试通过而删除大-inner 或某些 dtype case；这会隐藏真实数据损坏。
- 只比较 values 而忽略 indices；搬运破坏可能首先以无效索引出现。

## 误诊

- 失败跨四种 dtype 且集中在带余数的 PathB，不是某一 dtype 的比较精度或 NaN 规则问题。
- 开头 lane 正确不代表 scan carry 正确；该 trace 的损坏集中在 tail 搬运覆盖的列。