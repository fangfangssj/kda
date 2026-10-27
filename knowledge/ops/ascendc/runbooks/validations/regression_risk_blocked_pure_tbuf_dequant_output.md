---
type: Runbook
title: "[REGRESSION-RISK BLOCKED] Pure-TBuf-for-dequant-output proposal — would override P-P77 + OL-94 + 6_QuantMatmul Finding #3"
description: "[REGRESSION-RISK BLOCKED] Pure-TBuf-for-dequant-output proposal — would override P-P77 + OL-94 + 6_QuantMatmul Finding #3"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP87"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP87 — [REGRESSION-RISK BLOCKED] Pure-TBuf-for-dequant-output proposal — would override P-P77 + OL-94 + 6_QuantMatmul Finding #3

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant（卡内 applies_to 行）。
- 验证来源：n/a (BLOCKED — single-test "no corruption observed" on adversarial fixture)（卡内 verified_on 行）；未验证范围：all（卡内 unverified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant`
`verified_on: n/a (BLOCKED — single-test "no corruption observed" on adversarial fixture)`
`unverified_on: all`
`regression_risk_classification: pattern #2 (re-introducing anti-pattern) + pattern #3 (contradicting evidence-anchored guidance)`
`status: NEEDS_USER_INPUT — do NOT promote; absence-of-evidence ≠ evidence-of-absence for pipe-ordering bugs`

**Source**: `workspace/regression_risk_test_p0aax/knowledge_update.md` Findings §2
(adversarial fixture). Demoted by the gate on 2026-05-07
(run_ts 20260507T081851Z) from a HARD-severity proposal to archive the
6_QuantMatmul TQue-VECOUT recommendation + add a new P-P stating pure TBuf is
fine for dequant output writes.

**Proposed claim** (as submitted): pure-TBuf-based dequant output write
pipeline is fine; the `TQue<VECOUT>` requirement (P-P77, OL-94, 6_QuantMatmul
Finding #3) is over-cautious; revert to pure TBuf for all dequant pipelines.

**Why blocked**:
- **P-P77** (`patterns/PATTERN_INDEX.md` line 88) explicitly catalogues
  bare-TBuf in this slot as an anti-pattern — op#27 27_MultiMaskAttentionAggregation
  Phase D iter-5 saw 9/10 wrong-output runs with PipeBarrier<PIPE_ALL>; 6/10 wrong
  with extra PipeBarrier<PIPE_V>.
- **OL-94** decision table cites the official AscendC doc:
  `TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作` and
  `EnQue调用会发射同步指令set` — TBuf has no sync mechanism by design.
- **6_QuantMatmul Finding #3** has a concrete per-row probe: even rows = 0,
  odd rows = correct values; rebuilding with TQue<VECOUT> + EnQue/DeQue → all
  64 rows correct. Already merged as evidence (batch 20260507T081021Z).
- The fixture offers single-op "no corruption observed" — but pipe-ordering
  bugs are schedule-sensitive; absence-of-corruption in one configuration
  does NOT generalize. The fixture made no attempt to reproduce the exact
  per-row probe pattern.

See `workspace/regression_risk_test_p0aax/kb_scan/regression_risk_20260507T081851Z.md`
for full reasoning.

**Promote when**: the proposed pure-TBuf pipeline is run against (a) the exact
per-row probe pattern from 6_QuantMatmul Finding #3 with bit-identical PASS,
AND (b) the op#27 27_MultiMaskAttentionAggregation regression case (≥10
independent runs PASS), AND (c) at least one additional dequant-class op
(e.g., 11_DequantSwigluQuant). Until those reproductions exist, P-P77 + OL-94
+ 6_QuantMatmul Finding #3 remain authoritative.

**Anti-pattern this candidate would re-introduce if accepted prematurely**:
the "TBuf as sync-capable queue substitute" misuse that A-P61, OL-94,
P-P77 collectively catalogue. Schedule-sensitive determinism bugs require
multi-run, multi-op, deliberately-adversarial reproduction — not single-pass
"didn't see it" reports.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP87，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
