---
type: Runbook
title: "Multi-core data-movement partitioning — K-aligned row groups + DataCopy-only stores (NO scalar GM store)"
description: "dav-m200 多核搬运 kernel 的铁律是零标量 GM 写、全走对齐 DataCopy；推论是每个核的输出范围必须按 32B 块整数倍对齐切分（K 对齐行组）"
tags: [task.optimization, topic.data_movement.data_copy, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P277"]
---

# Multi-core data-movement partitioning — K-aligned row groups + DataCopy-only stores (NO scalar GM store)

## 优化点

(2026-08-29, pixel_shuffle/pixel_unshuffle 310p — multi-core companion to P-P276)
`applies_to: soc=Ascend310P3 (dav-m200, __NPU_ARCH__==2002); cann=9.0.0; op_class=data-rearrangement / data-movement (pure data movement, zero arithmetic)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (pixel_shuffle + pixel_unshuffle 2026-08-29: 8/8 contract each + large-shape maxerr=0 up to 4.19M; ~6-7× over single-core)`
`unverified_on: soc=Ascend950PR, Ascend910`
**Principle**: on dav-m200, a scalar-pipe GM `SetValue` store is **DROPPED under concurrency** —
even with per-core EXCLUSIVE 32B datablocks (PB-144, on-device bisect via block-census debug
kernels 3/3 repro; the group_norm_silu witness ships zero scalar GM stores). So a multi-core
data-movement kernel has ONE hard rule: **zero scalar GM stores — every output byte is written
by an aligned `DataCopy`.** The corollary: each core's output range must be a whole number of
32B blocks (single-writer-per-32B-block ownership), otherwise the boundary elements need a
scalar store and the race returns.
**Recipe** (builds on P-P276's three stages):
1. Keep P-P276's K-group store — `K = perBlock/gcd(Wout,perBlock)` so `K*Wout % perBlock == 0`.
2. Host tiling: cap `coreNum = min(GetCoreNumAiv(), totalRows / K)` and assign **K-aligned**
   row ranges (each core's `rowStart % K == 0` and `rowLen % K == 0`), so every core owns whole
   K-row groups = whole 32B blocks. The residual `totalRows % (coreNum*K)` rows go to ONE core.
3. Kernel: per K-row group → K `Gather`s into `outLocal[k*Wout]` + ONE `DataCopy(K*Wout)`
   (both sides 32B-aligned by construction). The residual (< K rows) uses P-P201's
   overlapping-32B-DataCopy-anchored-at-END (write ONE aligned 32B block ending at the region
   END, idempotently rewriting the last head lanes with identical bytes) — never a scalar GM
   write.
**Anti-pattern (what breaks it)**: a per-row scalar `outG.SetValue` fallback for non-aligned /
partial rows — under multi-core that fallback's stores vanish (output stays NaN-prefill at the
racy cores, non-deterministic which core drops). The single-core kernel may keep the scalar
fallback; the multi-core kernel MUST NOT.
**Concrete anchor**: `代码片段 pixel_shuffle__opsmath/pixel_shuffle.core.snippet.h（见文末）` +
`op_host/arch22/pixel_shuffle_tiling.cpp` (K-aligned coreNum cap).
- pixel_shuffle 310p (2026-08-29): K-aligned multi-core (coreNum = min(GetCoreNumAiv(), totalRows/K),
  group-based former/tail so every core owns whole K-groups) — 8/8 contract bit-exact + large-shape
  maxerr=0 (4.19M el / 512 rows); aligned 4.19M 648µs vs single-core 4785µs = **7.4×**.
- pixel_unshuffle 310p (2026-08-29): same recipe, 8/8 contract + large maxerr=0; 4.19M 974µs.
- The pre-fix failure mode (single-row tail cores → scalar fallback) reproduced exactly the
  PB-144 signature: racy cores' output stayed NaN-prefill, non-deterministic which core dropped.
- **Note**: `totalRows % K != 0` (residual < K rows) still needs the scalar fallback → multi-core
  unsafe → host falls back to `coreNum=1` for that rare shape; the DataCopy-only residual is the
  P-P201 overlap patch, not yet needed by the contract battery.
**Cross-ref**: [`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md) (overlapping-32B-END + PB-140), CAND-M200-CROSS-PIPE-32B-BLOCK-RACE,
P-P276 (the three-stage vectorization this layers on), P-P74 (host-precomputed partition
offsets). backend=ascendc.
- reduce_sum 310p (2026-08-17, 10th instance — PERF-side application, reduction op not
  elementwise): the ragged dim=0 path was a single-core scalar-GM-load loop (3861µs on
  [100,2007] fp16 vs 3.9µs vendor); re-expressing it as the head-tail-split two-pass
  (aligned bulk quanta + in-kernel ragged head/tail, no host staging) brought the case to
  106µs (+3550%). Confirms the recipe transfers beyond elementwise to reduction kernels
  whose raggedness is in the reduction INPUT extent. NOTE the multi-core scalar-STORE
  direction of this op hit PB-144 — loads reliable, stores not; output went all-MTE3
  instead of this pattern's scalar-store tail.
| P-P281 | **Silent-kernel write-behavior triage — dirty-allocator-block probe (unwritten vs garbage vs mis-addressed) + chrome-trace kernel-duration oracle (executed vs never-ran; duration-invariance localizes first-tile faults)** | debugging | kernel output is zero/garbage with NO error, or a "did it even run?" belief needs an execution oracle — required BEFORE theorizing on dav-m200, where OOB reads are silent (PB-154) | **HIGH** |

## 内嵌代码片段

### pixel_shuffle__opsmath/pixel_shuffle.core.snippet.h

```cpp

template <typename T>
__aicore__ inline void PixelShuffleCompute(GM_ADDR x, GM_ADDR y, const PixelShuffleTilingData* td)
{
    const int64_t useCoreNum = td->useCoreNum;
    const int64_t formerNum = td->formerNum;
    const int64_t formerLength = td->formerLength;
    const int64_t tailLength = td->tailLength;
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= useCoreNum) {
        return;
    }

    int64_t rowStart = 0;
    int64_t rowLen = 0;
    if (blockIdx < formerNum) {
        rowStart = blockIdx * formerLength;
        rowLen = formerLength;
    } else {
        rowStart = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
        rowLen = tailLength;
    }
    if (rowLen <= 0) {
        return;
    }
    const int64_t rowEnd = rowStart + rowLen;

    GlobalTensor<T> inG;
    GlobalTensor<T> outG;
    inG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t N = td->N;
    const int64_t Cin = td->Cin;
    const int64_t Hin = td->Hin;
    const int64_t Win = td->Win;
    const int64_t Cout = td->Cout;
    const int64_t Hout = td->Hout;
    const int64_t Wout = td->Wout;
    const int64_t r = td->r;
```
