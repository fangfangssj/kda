---
type: Runbook
title: "Multi-pass kernels — HBM bandwidth ceiling vs fused CANN"
description: "A multi-pass kernel reads each element from HBM N times (N=number of passes). A fused CANN kernel reads once. The performance ceiling of an N-pass kernel is ~1/N of fused, regardless of VEC optimizati"
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.memory, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-53-multi-pass-kernels-hbm-bandwidth-ceiling-vs-fused.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-53"]
---

# Multi-pass kernels — HBM bandwidth ceiling vs fused CANN

> 原 OKF v1 卡号：**OL-53**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - AscendC kernel performance <0.5x of CANN reference AND kernel requires multiple passes over input data

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
A multi-pass kernel reads each element from HBM N times (N=number of passes). A fused CANN kernel reads once. The performance ceiling of an N-pass kernel is ~1/N of fused, regardless of VEC optimization. For per-row reductions that need a global statistic before element-wise processing (e.g., quantization needs max before scaling), 2-pass is mandatory unless rows fit in single UB tile. When <0.5x vs CANN with a multi-pass kernel, check if the algorithm requires multi-pass before investing in VEC tuning — the bottleneck is HBM bandwidth, not compute.

## 证据
DynamicQuant 2-pass (find max, then quantize) → 0.25x of CANN's fused npu_dynamic_quant

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-53（category=performance_analysis，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
