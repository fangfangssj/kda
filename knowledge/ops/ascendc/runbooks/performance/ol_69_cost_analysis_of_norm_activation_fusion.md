---
type: Runbook
title: "Cost analysis of norm + activation fusion"
description: "In norm + activation fusion, the activation (e.g. Swish: `Muls(neg_x, x, -scale)` → `Exp` → `Adds(1)` → `Reciprocal` → `Mul`) adds only 5 VEC instructions to the normalize pass and **contributes almos"
tags: [task.optimization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.memory, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-69-cost-analysis-of-norm-activation-fusion.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-69"]
---

# Cost analysis of norm + activation fusion

> 原 OKF v1 卡号：**OL-69**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - planning to fuse norm (GroupNorm/LayerNorm/RmsNorm) + activation (Swish/GELU/SiLU) into a single kernel

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
In norm + activation fusion, the activation (e.g. Swish: `Muls(neg_x, x, -scale)` → `Exp` → `Adds(1)` → `Reciprocal` → `Mul`) adds only 5 VEC instructions to the normalize pass and **contributes almost nothing to end-to-end latency**. Reason: the kernel is bandwidth-bound on DataCopy, and the VEC instructions are hidden inside the pipeline. Fusion gain comes mainly from **eliminating the intermediate GM round-trip** (norm output and activation input are the same data).

## 证据
2_GroupNormSwish vs Pure PyTorch on the same-hardware baseline: 2.05x speedup; compared against standalone GroupNorm L1 at 0.76x, fusion yields ~2.7x net improvement (vs separate norm + separate activation). E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-69（category=performance，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
