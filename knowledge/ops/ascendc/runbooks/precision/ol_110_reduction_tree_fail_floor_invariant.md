---
type: Runbook
title: "Reduction-tree fail-floor invariant — swapping per-stage reduction shape only moves which cases fail, not the total"
description: "When the reference does cross-row aggregation on-NPU and the kernel decomposes it as a 2-stage SIMD reduction, swapping the per-stage shape (linear/tree/Kahan) shifts which cases fail MARE, not the total fail count."
tags: [task.debugging, topic.precision, topic.implementation.pipeline]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-110-reduction-tree-fail-floor-invariant.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
aliases: ["OL-110"]
---

# Reduction-tree fail-floor invariant — swapping per-stage reduction shape only moves which cases fail, not the total

> 原 OKF v1 卡号：**OL-110**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel writes Y = f(X).sum(dim=...) (fp32 many-element accumulation); reference is Model.forward returning torch_npu.<op> whose backend evaluates the same reduction on-NPU; verifier metric is MARE with threshold ~1e-3

## 现象 / 触发
Kernel output is `Y = f(X).sum(dim=(0,1))` (or an analogous many-element fp32 accumulation). The verifier reference is `Model.forward` returning `torch_npu.<op>(...)`, whose backend performs the same reduction internally on-NPU (not a CPU/fp64 ground truth). Metric is MARE (or any cancellation-sensitive relative error) with threshold ~1e-3. Precision-fix loops keep landing at the same residual fail count.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
When the reference does cross-row aggregation entirely on-NPU and the kernel decomposes the same aggregation as a 2-stage **per-block reduction → cross-block aggregation** SIMD pipeline, swapping the per-stage reduction shape (linear vs pairwise tree vs Kahan vs column-parallel-sequential vs binary-counter) tends to shift WHICH cases violate MARE rather than reduce the TOTAL fail count. For a fixed reference and shape set there is a **fail-floor invariant**: total failures stay roughly constant; only the identity of the failing cases moves.

This is distinct from OL-83 (single-ULP boundary tie) — the divergence here is many-element accumulation ordering, not a tie at one rounding boundary. It is the same "two valid fp32 paths" family but with a statistical MARE metric rather than bit-exact.

### Concrete anchor
19_FusedResidualRmsNormBackward kw-2 + kw-3 (2026-05-01), shape set 50, MARE thr=1.22e-3:
```
linear K1 + linear K2      → 47/50, fails {14, 23, 33}
pairwise tree K1 + linear  → 47/50, fails {14, 33, 43}   ← case 43 was passing at baseline
Kahan K1 + linear          → 46/50, fails {3, 14, 33, 43}
column-parallel direct     → 40/50, fails widely
```
Floor stays ~47/50 across single-knob algorithm changes; the pairwise-tree directive (probe-projected 48/50) measured 47/50 with case 43 as a NEW regression → REVERT.

### Recommended action when this signature applies
1. Implement the simplest baseline (linear–linear). Don't sweep K1/K2 reduction shapes hoping to close the residual.
2. Probe vs **fp64 truth** (not vs the torch_npu reference) to bisect the residual: kernel CLOSER to fp64 than torch_npu → an fp32 reduction-order residual (document against fp64 truth); kernel FURTHER from fp64 → genuine kernel issue, but a single algorithm swap typically just shifts the failure set (apply the OL-85 anti-overfit gate when comparing projected vs measured).
3. Ship at the fail-floor with all residuals classified.
