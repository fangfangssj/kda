---
type: Runbook
title: "Two-tier precision verdict with fail-closed independent-truth boundary"
description: "PASS_T1 and PASS_T2 remain grounded in declared CPU/source truth; a target-only differential is diagnostic and cannot become a final pass."
tags: [task.debugging, topic.precision]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-109-two-tier-precision-verdict.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
aliases: ["OL-109"]
---

# Two-tier precision verdict with fail-closed independent-truth boundary

> 原 OKF v1 卡号：**OL-109**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - Any op-archive verification at Phase O5 where per-dtype Tier-1 MERE/MARE thresholds are unreachable due to a hardware/algorithm precision floor (FMA grouping, hw sigmoid polynomial, parallel-reduction reorder)

## 现象 / 触发
Any op archive verification. Applies to ALL ops, but the Tier-2 fallback only fires for cases that fail Tier 1 — typically where the strict per-dtype threshold is unreachable due to a precision floor (FMA grouping, hardware sigmoid polynomial, parallel-reduction reorder). See `docs/design/PRECISION_METRICS_CANONICAL.md` for the full 2x2 grid of T1/T2 (judging standards) x Pass A/Pass B (input sets).

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
A single strict-vs-CPU threshold rejects kernels that are at the hardware precision floor yet still at least as good as the industry baseline. The rule is a ladder:

**Tier 1 — Strict (preferred):**
```
PASS_T1  ⟺  ours_MERE < threshold(dtype)  AND  ours_MARE < 10 × threshold(dtype)
```
All MERE/MARE computed against CPU truth (`Model.forward` on CPU). Per-dtype thresholds (same as production AscendC SKILL): fp16 = 2^-10 ≈ 9.77e-4, bf16 = 2^-7 ≈ 7.81e-3, fp32 = 2^-13 ≈ 1.22e-4. Integer / bool: bit-exact required — no Tier-2 fallback for ints.

**Tier 2 — Relative (fallback when T1 fails):**
```
PASS_T2  ⟺  ours_MERE ≤ CANN_MERE  AND  ours_MARE ≤ CANN_MARE
```
Both computed against CPU truth on the same case — i.e. ours is at least as accurate as CANN under the strict CPU-truth standard. CANN is the de-facto industry baseline; T2 asserts "we are not worse than the reference".

**Per-case verdict ladder (best → worst):**
| Verdict | Meaning |
|---|---|
| PASS_T1 | Tier 1 passes outright (independent of CANN); CPU+fp64 truth available. |
| PASS_T2 | Tier 1 fails, but ours ≤ CANN MERE/MARE (parity-or-better); CPU+fp64 truth available. |
| FAIL_NO_INDEPENDENT_TRUTH | Required CPU fp64 or selected-source-arch truth is unavailable. Target-NPU behavior may be recorded as a diagnostic, never as a final pass. |
| FAIL | T1 fails AND ours is strictly worse than the declared truth on ≥1 metric. |
| EVAL_ERR | Eval crashed / output-count mismatch / unknown-dtype — judge couldn't classify. |

### Target-only differential when CPU truth is structurally undefined
A target implementation may be run as prior-art or a diagnostic. It cannot establish final truth.
Restore the selected source-arch capture for migration or a CPU fp64 autograd oracle for backward;
otherwise emit `FAIL_NO_INDEPENDENT_TRUTH`.
