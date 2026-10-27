---
type: "Runbook"
title: "重复目标随机写丢失 — 在 UB 内按序合并后 DMA 写回"
description: "重复 overwrite 可在逐目标块 UB 内按序合并，但必须具备显式 TQue/事件依赖、每块唯一输出所有权和确定块间次序；否则仅作安全回退候选。"
tags: ["task.debugging", "topic.compute.indexing", "topic.implementation.memory_management", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["TQueBind", "explicit_dependency", "single_writer", "ordered_merge"]
---

# 重复目标随机写丢失 — 在 UB 内按序合并后 DMA 写回

## 现象

在 Ascend950 Flash 独立算子的 `operators/scatter/scatter.asc` 中实现 overwrite 语义时，kernel 先从 GM 逐元素读取索引和更新值，再直接对随机目标执行 `GlobalTensor::SetValue`。小规模重复目标用例可以通过，但放大后同一输入重复运行结果不稳定，并在目标位置和保留初值的位置出现非零 mismatch；构建始终成功，因此不是编译或 dtype 分发问题。

## 适用场景

- 工程形态：Flash kernel 直调，索引驱动的随机写；
- API 路径：`GlobalTensor::GetValue/SetValue`、`DataCopyPad`、`TQue` 与 `PipeBarrier`；
- 验证过的平台与版本：950（Ascend950PR_957b / dav-3510）+ CANN 9.0.1；
- 适用粒度：`operator_family`，适用于要求确定 overwrite 顺序的 scatter/index-put 类随机写；
- 路线优先级：`safety_fallback`。这是 fallback，性能不是最优，追求性能请先验证硬件原子、排序/分桶或其他目标聚合路线；
- 未验证场景：A2/A3、跨核允许任意 winner 的 overwrite 语义、超大目标空间下 gather 重扫的性能上限。

## 根因

失败路径把“读取 `idx/upd`”与“随机地址 `SetValue`”交错在同一内层循环，并对重复目标连续发出 GM 标量写；在 dav-3510 实测中，这个组合不能提供 overwrite 所需的稳定 last-wins 结果。把每个连续目标块先搬入 UB 后，全部重复更新都在 UB 内按原始 `k` 顺序合并，最后只以连续 DMA 写回 GM，便同时消除了随机 GM 标量写和重复目标的 winner 不确定性。

## 解决配方

先按连续输出地址划分每核责任区，再对责任区内的每个 UB chunk 执行：

1. `data[chunk]` 通过 `DataCopyPad` 搬入 UB；
2. 顺序扫描全部索引/更新 tile；
3. 命中当前 chunk 的更新在 UB 内执行 last-wins；
4. `PIPE_V` 排空 UB 标量/Vector 写后，用 32B 对齐 DMA 写回；
5. `PIPE_MTE3` 排空写回，再复用 UB；不足 32B 的末尾只写有效元素。

### before

代码证据：revision=`trace-captured pre-fix snapshot`；path=`operators/scatter/scatter.asc`；symbol=`KernelScatter<T, IdxT>::Process`；line span=`完整函数`。以下代码 verbatim 来自轨迹中的修复前函数写入：

```cpp
__aicore__ inline void Process()
{
    // Phase 1 (data->y copy) is done on the host via at::clone on the same
    // ACL stream (ordered before this kernel), so y already == data here.
    // Phase 2: scan all N updates; for each, compute target offset via the
    // odometer and write/RMW y[target] if it falls in this block's range.
    // idx/upd read via scalar GM loads (read-only -> multi-block safe).
    // Writes use scalar GlobalTensor::SetValue (element-granular, no 32B DMA
    // corruption). NOTE: scattered scalar stores lose ~0.01% of writes at
    // large scale (an AscendC store-buffer coherency quirk); the in-kernel
    // contiguous copy + cache-line-aligned partition keep this to a minimum.
    for (int64_t k = 0; k < N_; ++k) {
        int64_t idx = static_cast<int64_t>(idxGm_.GetValue(static_cast<uint64_t>(k)));
        int64_t target = base_ + idx * strideDim_;
        if (target < lo_ || target >= hi_) { advanceOdometer(); continue; }
        T u = updGm_.GetValue(static_cast<uint64_t>(k));
        if (reduceMode_ == RMODE_UPDATE) {
            yGm_.SetValue(static_cast<uint64_t>(target), u);
        } else if constexpr (!std::is_same_v<T, bfloat16_t>) {
            T cur = yGm_.GetValue(static_cast<uint64_t>(target));
            yGm_.SetValue(static_cast<uint64_t>(target), combineReduce<T>(reduceMode_, cur, u));
        } else {
            yGm_.SetValue(static_cast<uint64_t>(target), u);
        }
        advanceOdometer();
    }
}
```

### after

代码证据：revision=`2cc8abb (trace-provided abbreviated commit)`；path=`operators/scatter/scatter.asc`；symbols=`KernelScatter<T, IdxT>::Process`, `KernelScatter<T, IdxT>::GatherScatterOverwrite`；line span=`完整函数`。以下代码 verbatim 来自轨迹中的最终函数写入：

```cpp
__aicore__ inline void Process()
{
    // Overwrite or multiply on small/medium y: robust all-DMA gather-per-chunk
    // path. Each y chunk is DMA-loaded from data, RMW'd in UB (k-order =>
    // overwrite last-wins == golden; multiply combine is order-independent)
    // then DMA-stored. No scattered scalar GM store at all, so duplicate
    // targets incur zero loss. Large y stays on the scalar path: low
    // duplicate density at seed 0 and a gather re-scan of N per chunk would
    // be too slow for ~50M elements. (Reduce add/amax/amin use atomics, not
    // this path; fp16/bf16 reduce has no atomic and no eval dup case.)
    bool useGather = (reduceMode_ == RMODE_UPDATE && yNumel_ <= GATHER_OW_THRESHOLD) ||
                     (reduceMode_ == RMODE_MUL && yNumel_ <= GATHER_MUL_THRESHOLD);
    if (useGather) {
        GatherScatterOverwrite();
        return;
    }
    // Phase 1: data -> y. For overwrite/multiply (scalar Phase 2) the kernel
    // initialises y itself via DMA-load + contiguous scalar stores (no
    // GetValue during stores => 0 loss). For atomic reduce the host already
    // cloned data->y on the same stream, so Phase 1 is skipped.
    if (!initOnHost_) {
        CopyDataToY();
    }
    // Phase 2: scan all N updates. idx/upd are DMA-loaded into UB tiles, so
    // the scalar/atomic y writes happen with no concurrent GM load. Writes
    // use element-granular scalar SetValue (overwrite/multiply, in-order =>
    // last-wins matches golden) or hardware AtomicAdd/Max/Min (reduce,
    // order-independent & reliable). Duplicate targets are thus handled
    // correctly without any same-address store-buffer hazard.
    ScatterPhase();
}

__aicore__ inline void GatherScatterOverwrite()
{
    // Robust overwrite: per y chunk, DMA-load data[chunk] (y init), apply all
    // matching updates in UB (k-order last-wins == golden), DMA-store chunk.
    // All GM y writes are bulk DMA -> zero scattered-store loss.
    AscendC::DataCopyExtParams cpY{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> ppY{false, 0, 0, 0};
    AscendC::DataCopyExtParams cpIdx{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<IdxT> ppIdx{false, 0, 0, 0};
    AscendC::DataCopyExtParams cpUpd{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> ppUpd{false, 0, 0, 0};
    int64_t clElems = 32 / sizeof(T);
    if (clElems < 1) { clElems = 1; }
    for (int64_t chunkLo = lo_; chunkLo < hi_; chunkLo += CHUNK_ELEMS) {
        int64_t chunkLen = hi_ - chunkLo;
        if (chunkLen > CHUNK_ELEMS) { chunkLen = CHUNK_ELEMS; }
        // Load data[chunkLo, chunkLen] -> ubY (y initial value for this chunk).
        cpY.blockLen = static_cast<uint32_t>(chunkLen * sizeof(T));
        auto ubY = yChunkQue_.template AllocTensor<T>();
        AscendC::DataCopyPad(ubY, dataGm_[chunkLo], cpY, ppY);
        yChunkQue_.EnQue(ubY);
        ubY = yChunkQue_.template DeQue<T>();
        // Scan all N updates; apply those landing in this chunk to ubY.
        for (int32_t d = 0; d < ndim_; ++d) { pos_[d] = 0; }
        base_ = 0;
        int64_t k = 0;
        while (k < N_) {
            int64_t remain = N_ - k;
            int64_t n = remain < TILE_ELEMS ? remain : TILE_ELEMS;
            cpIdx.blockLen = static_cast<uint32_t>(n * sizeof(IdxT));
            cpUpd.blockLen = static_cast<uint32_t>(n * sizeof(T));
            auto ubIdx = idxQue_.template AllocTensor<IdxT>();
            auto ubUpd = updQue_.template AllocTensor<T>();
            AscendC::DataCopyPad(ubIdx, idxGm_[k], cpIdx, ppIdx);
            AscendC::DataCopyPad(ubUpd, updGm_[k], cpUpd, ppUpd);
            idxQue_.EnQue(ubIdx);
            updQue_.EnQue(ubUpd);
            ubIdx = idxQue_.template DeQue<IdxT>();
            ubUpd = updQue_.template DeQue<T>();
            for (int64_t i = 0; i < n; ++i) {
                int64_t idx = static_cast<int64_t>(ubIdx.GetValue(static_cast<uint64_t>(i)));
                int64_t target = base_ + idx * strideDim_;
                if (target >= chunkLo && target < chunkLo + chunkLen) {
                    uint64_t lo = static_cast<uint64_t>(target - chunkLo);
                    if (reduceMode_ == RMODE_UPDATE) {
                        ubY.SetValue(lo, ubUpd.GetValue(static_cast<uint64_t>(i)));
                    } else if constexpr (!std::is_same_v<T, bfloat16_t>) {
                        // multiply: UB read-modify-write (reliable in UB,
                        // order-independent for duplicate targets).
                        T cur = ubY.GetValue(lo);
                        T u = ubUpd.GetValue(static_cast<uint64_t>(i));
                        ubY.SetValue(lo, combineReduce<T>(reduceMode_, cur, u));
                    } else {
                        ubY.SetValue(lo, ubUpd.GetValue(static_cast<uint64_t>(i)));
                    }
                }
                advanceOdometer();
            }
            idxQue_.FreeTensor(ubIdx);
            updQue_.FreeTensor(ubUpd);
            k += n;
        }
        // Store ubY -> y[chunkLo, chunkLen]. Drain the scalar/V scatter writes
        // first (so the store DMA sees committed UB), and drain MTE3 after
        // (so the next chunk's load cannot overwrite this chunk's UB in
        // flight). 32B-aligned bulk via DMA; sub-32B tail via scalar so the
        // 32B write never spills past yNumel.
        AscendC::PipeBarrier<PIPE_V>();
        int64_t alignedLen = (chunkLen / clElems) * clElems;
        if (alignedLen > 0) {
            cpY.blockLen = static_cast<uint32_t>(alignedLen * sizeof(T));
            AscendC::DataCopyPad(yGm_[chunkLo], ubY, cpY);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        for (int64_t i = alignedLen; i < chunkLen; ++i) {
            yGm_.SetValue(static_cast<uint64_t>(chunkLo + i),
                          ubY.GetValue(static_cast<uint64_t>(i)));
        }
        yChunkQue_.FreeTensor(ubY);
    }
}
```

## 设计决策

- 选择 UB gather 是因为 overwrite 需要按原始更新次序保留 last-wins，`AtomicExch` 即使能写稳也不能保证 winner 与顺序语义一致；
- 放弃逐元素 `DataCopyPad`，因为 GM 单元素搬运会触及 32B 写粒度并可能破坏邻居；
- 当前方案属于 `safety_fallback`：正确性由 UB 内合并和连续 DMA 写回保证，代价是每个输出 chunk 都重扫索引。

## 性能影响

UB gather 的复杂度从“每核扫描一次更新流”变成“每输出 chunk 扫描一次更新流”；本轨迹未给出可跨算子泛化的相对性能比，因此只把它作为正确性兜底，不宣称性能收益。

## 预防

- 看到“编译通过 + 重复目标 + 每轮 mismatch 位置变化”时，先区分 overwrite winner 与普通精度误差；
- 随机 GM 写前先检查能否按目标排序、分桶、用硬件原子或在 UB 内聚合；
- UB 写回前后分别检查消费侧屏障与 MTE3 排空；尾块不得用 32B DMA 越过有效输出边界。

## 验证信号

来源是同一次 scatter 算子联调：先用设备诊断脚本复现重复目标 overwrite 的非确定 mismatch，再构建 UB gather 版本并重复执行诊断。

- 修复前重复目标 overwrite 出现非零 mismatch，且多次运行输出不一致（950，证据等级: tool_observed）；
- 修复后同一诊断输出 `mismatches=0`、`deterministic=True`（950，证据等级: tool_observed）；
- 最终干净构建后的 pytest 全部通过，完整设备精度评测全部用例通过（950，证据等级: tool_observed）。

## 失效边界

- 这是按目标空间分块重扫更新流的正确性路线；目标空间或更新流很大时可能接近超时，不应无条件作为 fast path；
- 只在 950 实测；A2/A3 未验证，推断可迁移但必须重新核验同步与 DMA 最小粒度；
- 如果语义允许任意 winner，可考虑更便宜的原子交换/分桶路线，但不得套用本卡的 last-wins 结论。

## 坏实践

- 继续在 `GetValue(idx/upd)` 与随机 `SetValue(y)` 交错的内层循环里加重试或经验性 barrier；这没有消除随机 GM 写本身；
- 用逐元素 `DataCopyPad` 模拟随机写；它把单元素语义误当成任意长度 memcpy；
- 通过调松 MARE/rtol 掩盖少量丢写。

## 误诊

最初把所有失败笼统归为“标量 GM store 在大规模必然丢写”。后续对照显示，无重复目标的置换路线可以稳定通过；真正需要优先规避的是重复目标下的随机写/RMW 与交错 GM load 组合。