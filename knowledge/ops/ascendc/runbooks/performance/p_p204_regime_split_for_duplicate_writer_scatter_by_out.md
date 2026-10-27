---
type: Runbook
title: "regime split for duplicate-writer scatter by output span M vs UB×cores"
description: "scatter 类按输出跨度 M 分两种打法：装得进 UB×核数就全暂存 UB，装不下就用 GM 驻留 fp32 delta + 每 32B 块单核所有权 RMW；delta 必须从 widen 基值初始化"
tags: [task.optimization, topic.compute.reduction]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P204"]
---

# regime split for duplicate-writer scatter by output span M vs UB×cores

## 优化点

**P-P204** — **regime split for duplicate-writer scatter by output span M vs UB×cores**: STAGE (M ≤ UB×cores ≈32768×8 on 310p: UB-resident output chunk, all reduce modes) vs DELTA (M beyond: output-partitioned rescanning costs O(passes×K) — ≥10 s at 141M elems / ~25-30 cyc/k-iter — so use a GM-resident fp32 delta: widen-init from x → per-hit 32B-block RMW → narrow once, reduce=add). Race-free without atomics: per-core ranges aligned ≥8 elems so every 32B fp32 block has ONE owner; three stream-ordered launches (OL-213, no SyncAll); `blk = off & ~7` into an 8-float TBuf. MANDATORY precision constraint: delta initialized from the widened base tensor, never zero ([`OL-547`](../precision/ol_547_a_gm_delta_rmw_design_must_initialize_the_delta.md)). 21_Scatter 310p 2026-08-15: 47/47 OVERALL_T2 + 501/501 + det bit-identical, M up to 141,557,760. Cross-ref [`OL-547`](../precision/ol_547_a_gm_delta_rmw_design_must_initialize_the_delta.md), P-P67, OL-213, [`OL-553`](../precision/ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md). backend=ascendc.

**Trigger classifier** (regime selection, by output span M vs UB×cores):
- STAGE regime — M ≤ UB×cores (≈32768×8 on 310p/dav-m200): stage the output chunk UB-resident, scan indices, write back once. Handles all reduce modes.
- DELTA regime — M beyond that: output-partitioned rescanning re-scans ALL K index elements per output chunk → O(passes×K) index traffic (measured ≥10 s on 141M-element cases at ~25-30 cyc/k-iter). Switch to a GM-resident fp32 delta: widen-init from x → per-hit 32B-block RMW → narrow once (reduce=add only).

**Technique** (dav-m200 anchor, `21_Scatter/kernel/scatter_kernel.h:537-661 [snippet-unavailable: source workspace not on this box]`): three STREAM-ORDERED launches (widen / RMW / narrow — no SyncAll, OL-213-safe). Per-core output ranges aligned to ≥ 8 elements so every 32B fp32 delta block has EXACTLY ONE owner core → race-free GM read-modify-write with the full MTE2→S→MTE3→MTE2 sync chain per hit, no `SetAtomicAdd`, deterministic by construction. Block RMW into an 8-float (32B) `TBuf` (`:579`): `blk = off & ~(int64_t)7` (`:661`) — load the 8-elem block, accumulate the hit, store back.

**Precision constraint (mandatory)**: the delta MUST be initialized from the widened base tensor, never zero — zero-init + post-add merge re-associates the per-slot fp chain and fails MARE-only T1 (OL-547).

**Evidence**:
- 21_Scatter 310p (2026-08-15, first instance): DELTA regime (reduce=add, M up to 141,557,760) + STAGE regime (all modes, M ≤ 262144); Pass A 47/47 OVERALL_T2, Pass B 501/501, determinism bit-identical, `probes/delta_bitprobe.py` 0/67M mismatches vs CPU-native. Sibling 19_IndexPut's output-partitioned design (M ≤ 131072) does not scale to 141M — the regime split is what keeps index traffic O(K).

**Other instances (predicted)**: embedding-backward large-vocab accumulate, `index_add_` beyond UB, any deterministic scatter-accumulate where atomics are barred by DET_POLICY or the no-atomic mandate.

**Cross-ref**: [`OL-547`](../precision/ol_547_a_gm_delta_rmw_design_must_initialize_the_delta.md) (the association rule this regime MUST obey), P-P67 (deterministic scatter-overwrite), OL-213 (stream-ordered launches, no SyncAll), [`OL-553`](../precision/ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md) (the graded-reference chain table), OL-252/OL-254 (determinism by construction). backend=ascendc.
