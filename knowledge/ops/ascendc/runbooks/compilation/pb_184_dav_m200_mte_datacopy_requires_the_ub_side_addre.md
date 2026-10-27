---
type: Runbook
title: "dav-m200 MTE `DataCopy` requires the **UB-side address** to be 32B-aligned (the GM side tolerates 4B granularity) — a tail block anchored at a row END touches UB at an unaligned offset and corrupts, and the `am`-anchor leaves `[am, H)` uncovered; ragged rows have NO valid DataCopy placement, emit them scalar"
description: "锚定在行尾的 DataCopy 尾块若 UB 侧非 32B 对齐会静默坏数据；改锚到对齐点又会漏掉尾段——尾块两头都要对齐。"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-184", "DataCopy"]
---

# dav-m200 MTE `DataCopy` requires the **UB-side address** to be 32B-aligned (the GM side tolerates 4B granularity) — a tail block anchored at a row END touches UB at an unaligned offset and corrupts, and the `am`-anchor leaves `[am, H)` uncovered; ragged rows have NO valid DataCopy placement, emit them scalar

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any kernel whose per-row element count is not a 32B-quantum multiple — ragged/variable-length rows)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (28_MultimodalRopePositionComputationWithGridBasedIndexing 310p, 2026-09-05, DEVICE-RUN device 3, H∈{8..128} row-length sweep)`
`unverified_on: soc=Ascend950PR (351x), Ascend910 (220x) — UB-side address-alignment behavior not probed there; do not import the constraint without a probe`
- **Symptom**: silent corruption (no trap). A `DataCopy` tail block anchored at the row END (`[H−qnt, H)`) touches the UB side at a non-32B-aligned offset → corrupt read/write. Re-anchoring the tail at the last aligned multiple `am` keeps UB aligned but leaves `[am, H)` uncovered — a missing-tail hole. The two placements fail in OPPOSITE directions; together with the 32B count quantum (PB-22) there is no DataCopy shape that covers a ragged row.
- **Constraint (device-probed)**: MTE `DataCopy` needs count = 32B quantum AND **UB-side address 32B-aligned**; the **GM side tolerates 4B granularity**. This is the UB-side sibling of PB-124's GM-side 32B-alignment corollary and CAND-DAV-M200-MULTICORE-ALIGN's GM partition-boundary rule.
- **Fix**: emit ragged rows (`rowElems % qnt != 0`) entirely scalar — scalar GM loads → compute → scalar GM stores, single core (PB-124); use `DataCopy` only for quantum-multiple rows where both sides are aligned by construction. Anchor (`代码片段 28_MultimodalRopePositionComputationWithGridBasedIndexing/mrope_kernel.h.snippet.h（见文末）`): "Ragged rows (rowElems % qnt != 0) must be emitted scalarly at the call site: any block copy at … ". Same shape as the P-P123 in-kernel scalar tail.
- **Evidence**: 28_MultimodalRopePositionComputationWithGridBasedIndexing 310p (2026-09-05): edge shapes H=127 / odd Q at iter-1 — row-END-anchored tail `DataCopy` corrupt AND `am`-anchored variant leaves `[am,H)` unwritten; full-scalar ragged path → pass_b edge cases clean, det 50/50 bit-identical (`28_MultimodalRopePositionComputationWithGridBasedIndexing/failures_ledger.md` precision iters 1-2).
- **Other instances (predicted)**: any dav-m200 kernel with variable row lengths — ragged-batch embed/gather/scatter, grid-tiled position tables, per-image variable-H layouts (DataCopyPad absent per PB-105, so tails are always hand-rolled).
- **Cross-ref**: [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (32B count-quantum family — incl. its dav-m200 sub-32B TRAP note), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (no DataCopyPad — why tails are hand-rolled), [`PB-124`](pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md) (GM-side alignment corollary + why the scalar tail must be single-core), CAND-DAV-M200-MULTICORE-ALIGN (GM-side partition alignment), P-P123 (the canonical scalar-tail pattern), EC-123 (scalar-vs-DMA same-line sibling). backend=ascendc.

## 内嵌代码片段

### 28_MultimodalRopePositionComputationWithGridBasedIndexing/mrope_kernel.h.snippet.h

```cpp
// bits [32,64) of the product = low32(mid) + high32(ll), with its carry into
// bit 64 — the original dumped high32(ll) into `hi`, corrupting P from bit 32
// up; python-diffed limb-by-limb on edge-18 args).
__aicore__ inline uint64_t MulWide(int64_t a, uint64_t b, uint64_t &hi) {
    const uint64_t av = static_cast<uint64_t>(a);
    const uint64_t a0 = av & 0xFFFFFFFFull, a1 = av >> 32;
    const uint64_t b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
    const uint64_t p3 = h0 + cy2 + cy3;
    // V2 = P * 2^k2 (P < 2^225, limbs p0..p3 little-endian).
    // Fraction limbs F = V2 mod 2^192:
    //   k2 >= 0 : F = (P mod 2^(192-k2)) << k2   — LOW bits of P, LEFT shift
    //             (the previous "P>>k2" was wrong: (P<<k2) mod 2^192 ≠ P>>k2;
    //             python-verified on edge case 18 args: >>13.3 rel vs <<1.5e-7)
    //   k2 <  0 : F = P >> |k2| (unreachable for |arg|>RED_MAX inputs — kept
    //             correct anyway)
    // of qnt (single DataCopy, both sides aligned). Ragged rows (rowElems %
    // qnt != 0) must be emitted scalarly at the call site: any block copy at
    // an in-row tail anchor either misses [am, H) or touches UB at unaligned
    // offsets (device-probed broken), and interleaving scalar tail writes
    // between MTE3 copies destabilises the pipeline (flaky leading-column
    // corruption, device-reproduced). Ragged + t>1 single-quantum overrun is
    // also unsound (frame-boundary clobber). Scalar GM SetValue everywhere.
    template <typename U>
    __aicore__ inline void StoreRow(GlobalTensor<U> &gm, int64_t elemOff, LocalTensor<U> src,
                                    int64_t rowElems, int64_t qnt) {
        DataCopy(gm[elemOff], src, static_cast<uint16_t>(rowElems));
    }

    __aicore__ inline void SyncSv() {
                Add(acc, acc, tmp, hh);
                if (H_ % QNT == 0) {
                    LocalTensor<half> ob = outQ_.AllocTensor<half>();
                    Cast(ob, acc, RoundMode::CAST_NONE, hh);  // single final RNE round
                    outQ_.EnQue(ob);
                    LocalTensor<half> od = outQ_.DeQue<half>();
                    for (int64_t f = 0; f < t; ++f) {
```
