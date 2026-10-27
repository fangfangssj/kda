---
type: Runbook
title: "Compute a dtype precision-floor by rounding the kernel's actual low-precision INPUT through the reference math — NOT by rounding the fp64 truth OUTPUT down to the dtype"
description: "Rounding the fp64 oracle OUTPUT down to the dtype captures only output-representation error and understates the true floor, flipping ceiling-vs-bug verdicts; feed the kernel's actual low-precision INPUT through the reference math and compare that to the fp64 oracle instead."
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-243-precision-floor-round-input-not-output.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-243"]
---

# Compute a dtype precision-floor by rounding the kernel's actual low-precision INPUT through the reference math — NOT by rounding the fp64 truth OUTPUT down to the dtype

> 原 OKF v1 卡号：**OL-243**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - Grading a low-precision (fp16/bf16) kernel as 'at the dtype ceiling' vs 'has a real bug', especially on an op that amplifies input perturbations (rope/rotary, small-denom normalization, recurrences, cancellation-prone reductions)

## 现象 / 触发
Grading a low-precision (fp16/bf16) kernel as "at the dtype ceiling" vs "has a real bug". `applies_to: soc=all; cann=all; op_class=all (precision-grading methodology)`. `verified_on: methodology validated on SFA (MLA rope) + selective_scan_source_a5, both A5`.

The wrong floor computation flips the verdict: an at-ceiling result reads as a bug, or a real bug reads as at-ceiling. This matters most when the op AMPLIFIES input perturbations (a small input rounding produces a large output error), because the input-rounding floor is then much larger than the output-rounding floor.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
The dtype-FLOOR you compare against MUST be computed by feeding the kernel's ACTUAL low-precision INPUT (the input tensors cast down to the kernel's dtype) through the reference math, then comparing that to the fp64 oracle. Rounding the fp64 TRUTH/oracle OUTPUT down to the dtype is WRONG: it captures only output-representation error and understates the true floor.

This is the input-side companion to the near-zero-cancellation metric-choice guidance (atol+rtol + matched-ratio for near-zero), but it is about WHAT you round, not which metric.

Concrete anchor (SFA, sibling op): a truth-rounded `bf16_floor` (~3.8e-3) made case-0 (MERE 2.91) look like a 766x bug -> the team concluded "rope bug". Re-computing the floor from the actual bf16-INPUT (MLA rope amplifies: an input perturbation -> MERE ~= 462) gave a true floor of ~2.31 -> case-0 at MERE 2.91 = 1.3x floor = AT CEILING = correct -> the rope-bug hypothesis was FALSIFIED purely by fixing the floor computation.

## 证据
- SFA precision reframe (2026-06-22): truth-rounded floor mis-calc -> false "rope bug"; input-cast floor -> case-0 at-ceiling, hypothesis falsified.
- selective_scan_source_a5 (2)(3)(4) (2026-06-22, A5): all ceiling-vs-bug verdicts computed with the input-cast floor.
- selective_scan_full_grad bwd 2.69x scan-vectorization (PR#37, `bda9cb3c`, 2026-06-22, A5): the post-opt 30/30 truth-backed precision verdict (fp32/fp16/bf16) was graded with the input-cast floor — the reverse-suffix Hillis-Steele rewrite (P-P106) is precision-NEUTRAL, so the at-ceiling verdict held before and after, confirming the floor methodology classifies a vectorization that does not move output as "no precision change".
- Cross-ref: P-P88 / OL-103 (primitive precision floors — a different floor source: the primitive, not the input rounding).
