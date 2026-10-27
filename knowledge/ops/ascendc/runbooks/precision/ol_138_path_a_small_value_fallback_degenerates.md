---
type: Runbook
title: "Path A canonical-evaluator §4.5.3 small-value fallback degenerates when the reference equals CPU truth"
description: "In port_a3_to_a5 Path A the §4.5.3 small-value fallback budget collapses to 2 absolute mismatches because cann_error_count is trivially 0 (CPU-truth vs CPU-truth) — too tight to absorb honest bf16/fp16 rounding; drive the kernel to bit-exact instead of loosening tolerance."
tags: [task.debugging, topic.precision, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-138-path-a-small-value-fallback-degenerates.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-138"]
---

# Path A canonical-evaluator §4.5.3 small-value fallback degenerates when the reference equals CPU truth

> 原 OKF v1 卡号：**OL-138**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - op runs in port_a3_to_a5 Path A (CPU-truth reference) with precision_eval_two_tier.py §4.5.3 small-value fallback enabled, and small-value outliers appear (e.g. bf16 case fails with 5 outliers, budget=2)

## 现象 / 触发

**Applies to** `soc=Ascend950PR; cann=9.0.0; op_class=any port_a3_to_a5 / CPU-truth-reference op`.
Verified on Ascend950PR / cann 9.0.0.

Loaded by aog-kernel-worker (Phase D, when reasoning about §4.5.3 fallback budgets) and
aog-precision-probe (when an op runs on Path A and small-value outliers appear).

**Trigger**: an op runs in `port_a3_to_a5` Path A (CPU-truth reference, because the vendor
torch_npu API is not exposed) AND the precision evaluator is the canonical two-tier
`precision_eval_two_tier.py` with the §4.5.3 small-value fallback enabled.

Concrete signature — ada_layer_norm Path A port (2026-05-13): iter-1 kernel produced 5
small-value outliers in bf16 case 5; the Path A §4.5.3 budget was 2 → FAIL.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

The §4.5.3 fallback rule allows `ours_error_count ≤ 2 × max(cann_error_count, 1)` for
small-value outputs where bit-exactness is implausible. It is calibrated for the standard
benchmark setup where `cann_error_count` reflects vendor-kernel small-value drift against CPU
truth — typically non-zero, giving our kernel headroom.

In Path A mode the "CANN" comparison degenerates to **CPU-truth vs CPU-truth** (no real vendor
kernel exists), so `cann_error_count = 0` trivially. The fallback budget collapses to
`2 × max(0, 1) = 2` absolute mismatches — far too tight to absorb the bf16/fp16 small-value
rounding that any honest kernel will produce. This degeneracy is **structural** (no second
reference exists to give headroom), not a bug.

### Recommended action when this signature applies

1. Treat §4.5.3 as effectively disabled in Path A — the evaluator is in
   "strict-bit-exact-or-very-close" mode.
2. Drive the kernel to literal-translation precision (P-P50, OL-112, OL-81) so element error
   counts drop to a handful at most.
3. Do NOT propose loosening tolerance or adding a Path-A-specific §4.5.3 expansion as a "fix";
   the degeneracy is structural, not repairable by widening the budget.
4. When reporting PARTIAL precision under Path A, distinguish a "true precision gap" from a
   "§4.5.3 would have rescued under Path B" case — the latter is informational, not a defense.

### Evidence

- ada_layer_norm (2026-05-13, A3→A5 kw-1..4): iter-1 had 7 cases PASS but bf16 case 5 hit
  MARE=1240 with 5 small-value outliers; §4.5.3 budget=2 left no headroom. Resolution was
  P-P50 (native-dtype post-modulation) → 8/8 PASS, 4 bit-exact. No evaluator change.

**Predicted other instances**: any future `port_a3_to_a5` op where Path A is forced (vendor
API absent on Ascend950PR); any custom-reference op where the evaluator's secondary reference
is the same as the primary CPU truth (CPU-truth-only benchmarks, hand-rolled reference ops).
(Source text truncated at this point.)
