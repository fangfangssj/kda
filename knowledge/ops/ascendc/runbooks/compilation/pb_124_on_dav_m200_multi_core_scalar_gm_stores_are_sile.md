---
type: Runbook
title: "on dav-m200, multi-core scalar GM stores are silently DROPPED under concurrency, even with per-core exclusive"
description: "dav-m200 多核标量 GM 写在并发下静默丢失（即使每核独占 32B 块）——多核输出必须走对齐 DataCopy"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-124"]
---

# on dav-m200, multi-core scalar GM stores are silently DROPPED under concurrency, even with per-core exclusive

**PB-124** — **on dav-m200, multi-core scalar GM stores are silently DROPPED under concurrency, even with per-core exclusive 32B datablocks** — 4 blocks scalar-storing to exclusive datablocks, one block's stores dropped 3/3 trials; not a flush delay; scalar GM LOADS unaffected. No fence fixes it (unlike the same-core, fenceable [`PB-123`](pb_123_on_dav_m200_scalar_gm_stores_sharing_a_64b_gm_se.md)): the only safe rule is zero multi-core scalar GM stores — single-writer-per-32B-block ownership, aligned DataCopy-only stores. Device-proven justification for [`P-P201`](../performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md)'s block-0-only scalar tail. Unaligned DataCopy dst works standalone but not under concurrency — keep all GM addresses 32B-aligned, loads included. group_norm_silu 310p 2026-08-17 (DEVICE-RUN 68/68, det 68/68). Cross-ref [`PB-123`](pb_123_on_dav_m200_scalar_gm_stores_sharing_a_64b_gm_se.md), [`P-P201`](../performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md), CAND-M200-CROSS-PIPE-32B-BLOCK-RACE, [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md). [dav-m200/Ascend310P3, multi-core, silent-data-loss]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (blockDim>1 kernels where MORE THAN ONE core issues scalar-pipe GM stores, e.g. SetValue tails staged per core)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (group_norm_silu 310p, 2026-08-17 — DEVICE-RUN, benchmark-acceptance lane; constant-input probes + block-census debug kernels, 3/3 repro)`
`unverified_on: soc=Ascend910 (220x), Ascend950PR (351x) — do not transfer without re-probe`

- **Severity**: HIGH — silent store loss + run-to-run non-determinism; launch and sync return clean, the dropped region keeps its pre-launch bytes.
- **Symptom**: with 4 blocks each scalar-storing (`GlobalTensor.SetValue`) to *exclusive* 32B datablocks (no sharing, no MTE3 involvement), one block's stores were dropped 3/3 trials; with shared datablocks, random subsets of stores were dropped per run. Stable after 0.5 s + full sync — NOT a flush delay. Scalar GM **loads** are unaffected (multi-core gamma/beta `GetValue` loads verified across all 68 cases).
- **Scope clarification (vs PB-123 — read both before choosing a fix)**: PB-123 is a SAME-core ordering hazard (scalar store sharing a 64B sector with an in-flight MTE3 copy) that a `SetFlag/WaitFlag<HardEvent::MTE3_S>` fence DOES fix. PB-124 is a CROSS-core concurrency hazard on the scalar store path itself — no fence fixes it; the only safe rule is **no scalar GM stores from multiple cores, period**. Single-writer scalar stores remain OK: this PB is the device-proven reason P-P201's "block-0-only scalar tail" convention exists (one core scalar-stores, the rest use aligned DataCopy), upgrading that convention to a necessity. It also explains the PB-106-adjacent "multi-core scalar" folklore (CAND-DAV-M200-SCALAR-READ-STALE-MULTICORE is the read-side sibling).
- **Fix (shipped recipe)**: zero scalar GM stores in multi-core kernels. Single-writer-per-32B-block ownership: every GM output block is written by exactly one aligned `DataCopy`; ragged row head/tail handled by aligned loads + UB lane-picking, never by element-wise `SetValue`. Corollary on alignment: an element-unaligned `DataCopy` dst works STANDALONE on dav-m200 (probe: dst elem-offset 4 fp32 = byte 16 landed 32/32 values) but must NOT be trusted under concurrency (v1's unaligned stores + concurrent scalar stores to the same 32B block lost data) — keep ALL DataCopy GM addresses 32B-aligned, loads included.
- **Detection rule for future ops**: multi-core m200 kernel with per-core scalar GM stores + flaky large-diff holes that move across reruns → this class. Bisect with constant-input probes (each core stores a distinct constant to its exclusive region, census what landed) before theorizing about fences.
- **Concrete anchor**: `代码片段 group_norm_silu/group_norm_silu_kernel.h.snippet.h（见文末）` (v2 shipped form — block-ownership partition, aligned DataCopy-only stores; workspace scratch, gitignored).
- **Evidence**: group_norm_silu 310p (2026-08-17, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): v1 (scalar GM store tails + element-unaligned DataCopy dst) failed det with large/garbage diffs on 41/68 cases; on-device bisect isolated the multi-core scalar-store drop (exclusive-datablock probe, 3/3). v2 rewrite (zero scalar GM stores, single-writer ownership) → 67/68; remaining case-63 nondeterminism was the separate staged-buffer fence class (CAND-DAV-M200-STAGED-BLOCK-FENCE-SET) → 68/68 PASS, det 68/68 bit-identical, max_abs_diff 3.9e-3 (fp16 ULP), 6/68 bit-exact (`group_norm_silu/failures_ledger.md` precision iters 1-2, `group_norm_silu/verification.json`).
- **Other instances (predicted)**: any multi-core dav-m200 kernel tempted to use `SetValue` for ragged tails, per-core variable-length outputs, scalar scatter/epilogues — scatter tails, string/categorical encoders, ragged-batch norm reductions, per-core count-prefix writes.
- **Cross-ref**: [`PB-123`](pb_123_on_dav_m200_scalar_gm_stores_sharing_a_64b_gm_se.md) (the same-core MTE3↔scalar-store ordering sibling — fenceable, different fix), [`P-P201`](../performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md) (block-0 scalar tail convention — this PB is its justification), CAND-M200-CROSS-PIPE-32B-BLOCK-RACE (cross-core S-vs-MTE3 tear on a SHARED 32B block — PB-124 drops stores even on EXCLUSIVE blocks), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (no DataCopyPad — why tails are tempting here), (read-side sibling). backend=ascendc.

- **Evidence (2nd op, independent probe)**: reduce_sum 310p (2026-08-17, CANN 9.1.0) — dedicated multi-core probe reproduced the drop at blockDim 4/8 (~50% of disjoint 32B-separated scalar stores lost, run-varying, silent rc==0) while blockDim ≤ 2 landed reliably in the same probe; multi-core scalar LOADS 100% reliable (store-direction-only, matching above). Additional data points: `SetFlag/WaitFlag<HardEvent::S_MTE3>` + trailing dummy MTE3 copy does NOT fix it, and `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` is an UNDEFINED SYMBOL on m200 (declared, not implemented — PB-105/PB-110 missing-API class). Shipped fix: all output via UB + vector DataCopy + host quantum-multiple over-alloc/narrow; 37/37 tier-1 PASS.

## 内嵌代码片段

### group_norm_silu/group_norm_silu_kernel.h.snippet.h

```cpp
//   => fp32 pipeline is DS end-to-end:
//     pass 1: TwoSum-compensated lane sum of x           -> mean = DS / L
//     pass 2: d = x - mean_h; TwoSum-compensated sum d^2  -> var  = DS / L
//     rstd: DS Newton rsqrt from vector-Sqrt seed         (~1e-13 rel)
//     per segment: c1 = DS(rstd)*gamma, c2 = beta - c1*DS(mean) (scalar DS)
//     y = (x*c1h + c2h) + (x*c1l + c2l) with exact TwoProd/TwoSum residuals
//   DS residual error ~2^-45 relative — 4+ orders inside the compare.py
//   thresholds; verified against the vendored grader in sim_numerics.py.
//
// Algorithm skeleton (V220 g32 ancestor): per group row r (contiguous flat
// span [r*L,(r+1)*L), L=(C/G)*HW); y=(x*scale+bias), scale=rstd*gamma[c],
// bias=beta[c]-scale*mean (V220 fold, mirrors PyTorch CPU apply loop).
// fp16: fp32 compute; CAST_NONE (RNE on dav-m200, EC-104) both directions.
// No Compare/Select (PB-66), no ShiftLeft (PB-64), no DataCopyPad (PB-59),
// Adds(-s) instead of Subs (EC-93), count-form VEC ops (EC-96),
// WholeReduceSum single-repeat int32-mask (fp16 path, OL-347), full

private:
    // ---- double-single (float-float) scalar primitives ---------------------
    // TwoSum: exact a + b = s + e (branchless, 6 flops).
    __aicore__ inline void TwoSumSc(float a, float b, float& s, float& e)
    {
        s = a + b;
        float bp = s - a;
        e = (a - (s - bp)) + (b - bp);
    }

    // Veltkamp split: v = vh + vl exactly, vh/vl non-overlapping halves.
    __aicore__ inline void VeltkampSc(float v, float& vh, float& vl)
    {
        float t = v * 4097.0f;  // 2^12 + 1
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(ev);
        WaitFlag<HardEvent::MTE3_S>(ev);
    }

    __aicore__ inline void FenceSMTE3()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ev);
        WaitFlag<HardEvent::S_MTE3>(ev);
    }

    // ---- scalar helpers ---------------------------------------------------
    __aicore__ inline float LoadScalarGB(const GlobalTensor<T>& gm, int64_t idx)
    {
```

**孪生卡 pb-176 合并的独有内容**（pb-176-on-dav-m200-multi-core-scalar-gm-stores-are-sile.md 已删除）：

> - **Scope clarification (vs PB-123 — read both before choosing a fix)**: PB-123 is a SAME-core ordering hazard (scalar store sharing a 64B sector with an in-flight MTE3 copy) that a `SetFlag/WaitFlag<HardEvent::MTE3_S>` fence DOES fix. PB-124 is a CROSS-core concurrency hazard on the scalar store path itself — no fence fixes it; the only safe rule is **no scalar GM stores from multiple cores, period**. Single-writer scalar stores remain OK: this PB is the device-proven reason P-P123's "block-0-only scalar tail" convention exists (one core scalar-stores, the rest use aligned DataCopy), upgrading that convention to a necessity. It also explains the PB-106-adjacent "multi-core scalar" folklore (CAND-DAV-M200-SCALAR-READ-STALE-MULTICORE is the read-side sibling).

> - **Concrete anchor**: `target/ascendc/examples/group_norm_silu/snippets.md` (v2 shipped form — block-ownership partition, aligned DataCopy-only stores; workspace scratch, gitignored).

> - **Evidence (2nd op — scan kernel; unaligned-layout compliant fallback shape)**: cumsum 310p (2026-08-18, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): greenfield multi-core scan kernel — 3 of the 4 initial store paths violated this PB (per-core scalar GM stores for ragged scan-line boundaries) before the rule was applied. Shipped compliant shape for UNALIGNED layouts (inner%quantum!=0, scan%quantum!=0): flat-element aligned partition + per-line prefix recompute via scalar GM **loads** (legal — the read side is unaffected) + staged aligned `DataCopy` stores + sole-owner RMW of the single trailing partial 32B block. Acceptance precision 42/42 PASS on the CPU-truth lane (`cumsum/failures_ledger.md` precision iter 1, `cumsum/verification.json`). Generalized as candidate CAND-M200-FLAT-PARTITION-TAIL-RMW.

> - **Evidence (3rd op — shared-32B-line lost updates under ragged row starts; nblk=1 clamp for the ragged surface only)**: 28_MultimodalRopePositionComputationWithGridBasedIndexing 310p (2026-09-05, benchmark, DEVICE-RUN device 3): ragged output rows (H%32B≠0 / 4Q%qf≠0) begin at 4B-aligned GM offsets, so scalar `SetValue` emission from multiple cores shared 32B lines → line-buffered lost updates in 8-element head/tail corruption groups (3-image repro Q=31/H=127); corruption eliminated by forcing nblk=1 whenever the output layout is ragged — the benchmark surface (all 50 cases H%8==0 ∧ Q%2==0, quantum-aligned) keeps the 8-core VEC path untouched (no OL-397 violation: scalar/single-core confined to EDGE shapes outside the perf surface). The 8-element (=32B fp32-scale) corruption grouping is a data point for CAND-M200-SCALAR-STORE-XCORE-512B-TEAR's open tear-granularity question. `28_MultimodalRopePositionComputationWithGridBasedIndexing/failures_ledger.md` precision iter 2.

> - **Cross-ref**: [`PB-123`](pb_123_on_dav_m200_scalar_gm_stores_sharing_a_64b_gm_se.md) (the same-core MTE3↔scalar-store ordering sibling — fenceable, different fix), P-P123 (block-0 scalar tail convention — this PB is its justification), CAND-M200-CROSS-PIPE-32B-BLOCK-RACE (cross-core S-vs-MTE3 tear on a SHARED 32B block — PB-124 drops stores even on EXCLUSIVE blocks), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (no DataCopyPad — why tails are tempting here), (read-side sibling). backend=ascendc.
