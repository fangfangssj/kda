---
type: Runbook
title: "regime split for duplicate-writer scatter by output span M vs UB×cores"
description: "regime split for duplicate-writer scatter by output span M vs UB×cores"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P204 — P-P204: A5Ops additive extension

> 关联：同一 P-P204 主题已由既有卡 [p_p204_regime_split_for_duplicate_writer_scatter_by_out.md](p_p204_regime_split_for_duplicate_writer_scatter_by_out.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文 Trigger classifier（按输出跨度 M vs UB×cores 选择 STAGE/DELTA regime，锚定 310p/dav-m200）。
- 验证来源：卡内无 verified_on/confirmed_on 行；验证证据见正文 Evidence 行（21_Scatter 310p，2026-08-15）。

**P-P204** — **regime split for duplicate-writer scatter by output span M vs UB×cores**: STAGE (M ≤ UB×cores ≈32768×8 on 310p: UB-resident output chunk, all reduce modes) vs DELTA (M beyond: output-partitioned rescanning costs O(passes×K) — ≥10 s at 141M elems / ~25-30 cyc/k-iter — so use a GM-resident fp32 delta: widen-init from x → per-hit 32B-block RMW → narrow once, reduce=add). Race-free without atomics: per-core ranges aligned ≥8 elems so every 32B fp32 block has ONE owner; three stream-ordered launches (OL-213, no SyncAll); `blk = off & ~7` into an 8-float TBuf. MANDATORY precision constraint: delta initialized from the widened base tensor, never zero (OL-547). 21_Scatter 310p 2026-08-15: 47/47 OVERALL_T2 + 501/501 + det bit-identical, M up to 141,557,760. Cross-ref [OL-547](../precision/ol_547_a_gm_delta_rmw_design_must_initialize_the_delta.md), [P-P67](../../optimizations/p_p67_pytorch_ub_class_scatter_overwrite_npu_reference.md), [OL-213](../../optimizations/ol_213_aicore_only_pipebarrier_not_syncall.md), [OL-553](../precision/ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md). backend=ascendc.

- Former A5Ops ID: `P-P204`
- Fusion disposition: `additive_new`

**Trigger classifier** (regime selection, by output span M vs UB×cores):
- STAGE regime — M ≤ UB×cores (≈32768×8 on 310p/dav-m200): stage the output chunk UB-resident, scan indices, write back once. Handles all reduce modes.
- DELTA regime — M beyond that: output-partitioned rescanning re-scans ALL K index elements per output chunk → O(passes×K) index traffic (measured ≥10 s on 141M-element cases at ~25-30 cyc/k-iter). Switch to a GM-resident fp32 delta: widen-init from x → per-hit 32B-block RMW → narrow once (reduce=add only).

**Technique** (dav-m200 anchor, 内部路径（不公开）workspace/21_Scatter/kernel/scatter_kernel.h:537-661): three STREAM-ORDERED launches (widen / RMW / narrow — no SyncAll, OL-213-safe). Per-core output ranges aligned to ≥ 8 elements so every 32B fp32 delta block has EXACTLY ONE owner core → race-free GM read-modify-write with the full MTE2→S→MTE3→MTE2 sync chain per hit, no `SetAtomicAdd`, deterministic by construction. Block RMW into an 8-float (32B) `TBuf` (`:579`): `blk = off & ~(int64_t)7` (`:661`) — load the 8-elem block, accumulate the hit, store back.

**Precision constraint (mandatory)**: the delta MUST be initialized from the widened base tensor, never zero — zero-init + post-add merge re-associates the per-slot fp chain and fails MARE-only T1 (OL-547).

**Evidence**:
- 21_Scatter 310p (2026-08-15, first instance): DELTA regime (reduce=add, M up to 141,557,760) + STAGE regime (all modes, M ≤ 262144); Pass A 47/47 OVERALL_T2, Pass B 501/501, determinism bit-identical, `probes/delta_bitprobe.py` 0/67M mismatches vs CPU-native. Sibling 19_IndexPut's output-partitioned design (M ≤ 131072) does not scale to 141M — the regime split is what keeps index traffic O(K).

**Other instances (predicted)**: embedding-backward large-vocab accumulate, `index_add_` beyond UB, any deterministic scatter-accumulate where atomics are barred by DET_POLICY or the no-atomic mandate.

**Cross-ref**: OL-547 (the association rule this regime MUST obey), P-P67 (deterministic scatter-overwrite), OL-213 (stream-ordered launches, no SyncAll), OL-553 (the graded-reference chain table), OL-252/OL-254 (determinism by construction). backend=ascendc.
