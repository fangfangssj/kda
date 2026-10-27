---
type: Runbook
title: "At a finalize gate that re-measures claimed counts, emit a concrete pass count for EVERY pass the runner can reliably reproduce — minimize N/A claims so one transient measurement can't sink a correct archive"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.data_movement, topic.performance.analysis, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-218-at-a-finalize-gate-that-re-measures-claimed-counts.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-218"]
---

# At a finalize gate that re-measures claimed counts, emit a concrete pass count for EVERY pass the runner can reliably reproduce — minimize N/A claims so one transient measurement can't sink a correct archive

> 原 OKF v1 卡号：**OL-218**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - <!-- applies_to_backend: all -->

<!-- applies_to_backend: all -->
`applies_to: soc=all (harness/finalize policy — chip-independent); cann=all; bisheng=n/a; op_class=all; mode=all`

**Principle**: when a finalize/post-verify gate independently re-measures the pass counts an archive claims (O5-style claim-vs-measurement reconciliation), every N/A or single-source claim is a fragile point: a one-off transient on that measurement (shared-lane NPU contention, a `None` read in a canonical→fallback chain) produces a claim-vs-measurement discrepancy and rolls back an otherwise-correct archive. **Mitigation: leave the FEWEST fragile/N-A claims** — emit a concrete count for every pass the gate's runner can reliably reproduce (e.g. promote a deterministic-subset `pass_b` from N/A to its real count), so the verification.json claim surface fully matches what the runner re-measures and a single transient on one measurement degrades to a 1-of-2 confirmation instead of a total rollback.

**Concrete anchor**: a finalize re-measure read `pass_a=None` during one run (transient shared-lane contention) → "1 discrepancy" rollback even though an independent re-run of the EXACT O5 path gave 48/48 in 26s. Fix: promote `pass_b` from N/A to a concrete **15/15 PASS** (the `pass_b_runner` reliably re-measures the deterministic no-drop subset bit-exact), so verification.json claims BOTH passes the runner can reproduce → 2/2 VERIFIED.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Evidence
- FA-A5 `3_FusionAttention` 2026-06-11 (kw-3, O5 transient close-out): diagnosed via Probe-Before-Fix (re-ran `pass_a_runner.py` under O5's exact docker invocation → 48/48 PASS, no timeout/crash; re-ran the exact `phase_o5.post_verify_for_finalize` path → VERIFIED). Robustness fix promoted pass_b N/A→15/15 with no kernel edit; re-ran O5 → VERIFIED, mismatches=[].

## Other instances (predicted)
- Any harness with a measurement-reconciliation finalize gate (port_a3 pass_a/pass_b, benchmark dual-count, any future mode). Any op with a deterministic subset that can carry a concrete reproducible count alongside a non-deterministic / replay-gated subset.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-218（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
