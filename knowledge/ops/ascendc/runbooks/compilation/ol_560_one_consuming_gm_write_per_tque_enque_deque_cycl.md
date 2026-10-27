---
type: Runbook
title: "One consuming GM write per TQue EnQue/DeQue cycle — a dequeued tensor held across multiple DataCopy issues silently drops all but the first few writes; the per-cycle cost (~0.5µs) sets the batching floor, not the ceiling"
description: "一次 TQue EnQue/DeQue 周期只覆盖一次 MTE3 消费写；拿着出队的 tensor 发多次 GM 写会静默丢弃后面的写——解法不是 fence 是纪律（每周期一写）。"
tags: [task.debugging, topic.data_movement.data_copy, topic.implementation.synchronization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-560"]
---

# One consuming GM write per TQue EnQue/DeQue cycle — a dequeued tensor held across multiple DataCopy issues silently drops all but the first few writes; the per-cycle cost (~0.5µs) sets the batching floor, not the ceiling

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any VECOUT UB→GM write path)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (22_HybridAttentionMaskPreparation kw-1 iter 1, 2026-09-05 — device run; 50/50 pass_a/pass_b bit-exact + 50/50 3-run determinism on the fixed kernel)`
`unverified_on: soc=Ascend950PR, Ascend910_V220 (queue-slot contract is arch-general in principle and 13_Cat/16_Repeat ship the same discipline on dav-m200, but no A5/A3 instance or counter-instance either way)`

**Principle**: a `TQue<VECOUT>` tensor's queue contract covers exactly ONE consuming MTE3 `DataCopy` per Alloc→VEC-op→EnQue→DeQue→FreeTensor cycle. Holding one dequeued LocalTensor and issuing MANY GM writes from it puts the extra MTE3 issues outside the queue's event protocol — on dav-m200 only the first ~2-4 writes land; the rest vanish with NO error code (silent drop, not corruption, not intermittent). The fix is not a fence — it is the discipline itself: one full queue cycle per GM write. This is the pure-write sibling of PB-11's loop-class DataCopy corruption and of CAND-DAV-M200-MULTIPART-RMW-INTERLOCK-TRIO's race point (c) ("canonical EnQue/DeQue protocol on every VECOUT MTE3-out").

**Queue-cycle economics (why you batch INSIDE one cycle, never by reusing slots)**: one Alloc→VEC→EnQue→DeQue→DataCopy→Free cycle costs ~0.5µs wall on dav-m200. A row-granular write scheme (2-3 cycles per row) measured 0.29× mean vs the torch reference — far below any bar in effect (kw-1 cited a "0.6× gate" here; that bar was worker-authored and is superseded by the owner parity bar 1.0 — see the reconciliation note under Evidence). Amortize by writing BIGGER per-cycle extents (broadcast one pattern plane per core, then 32KB-chunked GM→UB→GM copies at 2 cycles per 32KB), never by issuing more DataCopys per cycle: the same op lifted to 0.69× mean, [`OL-95`](ol_95_when_the_reference_path_s_output_depends_on_rng_gl.md) profiler-free wall-clock 1.77× on the largest case. Cross-ref OL-63 (tile-first/depth-second) — that rule picks depth for NON-overlapping throughput streams; it does not license slot reuse for extra writes.

**Concrete anchor** (per-write cycle, depth-1 queue — `22_HybridAttentionMaskPreparation/kernel/hybrid_mask_kernel.h:13-24,118 [snippet-unavailable: source workspace not on this box]`, depth const `22_HybridAttentionMaskPreparation/kernel/hybrid_mask_tiling.h:18 [snippet-unavailable: source workspace not on this box]`):
```cpp
pipe_.InitBuffer(outQ_, 1, bufBytes_);                 // depth 1
auto w = outQ_.AllocTensor<T>();                       // one cycle per GM write
Duplicate(w.ReinterpretCast<uint32_t>(), ones ? 0x01010101u : 0u, count >> 2);
outQ_.EnQue(w); auto wd = outQ_.DeQue<T>();
DataCopy(gm[off], wd, count);                          // the ONE consuming write
outQ_.FreeTensor(wd);
```

- **Evidence**: 22_HybridAttentionMaskPreparation kw-1 iter 1 (2026-09-05, Ascend310P3/dav-m200, CANN 9.0.0, benchmark — hybrid full+SWA attention-mask materialization): initialized two constant UB tensors once from a single DeQue and issued many DataCopys — only the first ~2-4 writes per mask landed (`22_HybridAttentionMaskPreparation/failures_ledger.md:7`); the per-write cycle shipped → 50/50 pass_a OVERALL_T1 bit-exact, 50/50 pass_b bit-exact, det 50/50 3-run identical, perf 0.6915× mean (`22_HybridAttentionMaskPreparation/verification.json`). **Perf-bar reconciliation (kw-3, 2026-09-05, audit F2/F5)**: that verification.json now records `threshold: 1.0` (owner-directed paradigm-uniform parity bar 2026-07-21, `base.py:405-406`（a5_ops 路径，未迁移）; kw-1's 0.6 was the pre-2026-07-21 default, self-authored into the artifact) and the kw-2 P149-remediated binary at 0.8648× mean / 0.8395× median → status FAIL vs 1.0 (39/50 cases < 1.0); **ko-2 update (2026-09-06, Opt1)**: the batching this entry prescribes was then applied — band-tiled generation + ONE consuming write per band per plane (UB multicast, `CAND-DAV-M200-BAND-TILED-UB-MULTICAST`) — lifting the same op to 1.3258× mean / 1.1946× median (43/50 ≥ 1.0; pass_a/pass_b + det re-verified 50/50 on the same binary) → status PASS vs 1.0. **Final reconciliation (kb-merge 2026-09-06; capture at the 2026-09-05T17:07Z finalize)**: the previously deferred orchestrator-side independent capture HAS run — `independent_re_measure.ran=true` in `22_HybridAttentionMaskPreparation/verification.json` (`phase_o5_perf_capture`: torch_npu.profiler, symmetric reference-vs-candidate, orchestrator's own context, independent of the worker self-report) — device-self-duration ratio **8.0841×** (reference 207.953µs vs candidate 25.724µs), now the artifact's top-level `performance.ratio` with the worker's 1.3258× wall-clock 50-case mean preserved verbatim under `worker_authored_aux`; the two figures are different methodologies (profiler device-time on the capture config vs wall-clock case-mean), not a contradiction, and PASS vs the owner 1.0 bar holds under both. The 0.29×/0.69×/1.77× figures in this entry are kw-1 device measurements retained as the batching-economics anchor, NOT the current verdict. Device run — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) still carries the stale pre-device `device.runtime: NOT_RUN` flag; the run artifacts supersede it here (PB-74 precedent).
- **Other instances (predicted)**: mask/pattern materialization and constant-fill broadcast kernels; multi-segment emit kernels tempted to hoist one DeQue out of the loop (scatter/gather staging, repeat/tiling broadcast emits); any kernel that "optimizes" queue overhead by reusing a dequeued tensor for several stores.
- **Cross-ref**: [`PB-11`](pb_11_datacopy_to_tbuf_veccalc_silent_corruption_on_mult.md) (loop-class DataCopy corruption), [`OL-25`](ol_25_tbuf_has_no_automatic_synchronization_must_use_tqu.md)/OL-94 (queue vs TBuf sync contracts), CAND-DAV-M200-MULTIPART-RMW-INTERLOCK-TRIO (race point (c) — same protocol requirement in RMW form), CAND-DAV-M200-VECOUT-MTE2-MTE3-FLAGS (what the queue does NOT order), OL-397 (vectorized+multi-core routing the batching enables). backend=ascendc.
