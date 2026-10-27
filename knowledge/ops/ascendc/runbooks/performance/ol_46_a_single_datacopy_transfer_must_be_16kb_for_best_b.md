---
type: Runbook
title: "A single DataCopy transfer must be ≥16KB for best bandwidth"
description: "Single-transfer sizes <16KB have significantly reduced bandwidth utilization. Grow TILE_SIZE as large as possible so each DataCopy moves ≥16KB. fp32 TILE=4096 (16KB), fp16/bf16 TILE=8192 (16KB) are re"
tags: [task.optimization, topic.data_movement.data_copy, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-46-a-single-datacopy-transfer-must-be-16kb-for-best-b.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-46"]
---

# A single DataCopy transfer must be ≥16KB for best bandwidth

> 原 OKF v1 卡号：**OL-46**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - tile-size selection for DataCopy in a SIMD kernel

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Single-transfer sizes <16KB have significantly reduced bandwidth utilization. Grow TILE_SIZE as large as possible so each DataCopy moves ≥16KB. fp32 TILE=4096 (16KB), fp16/bf16 TILE=8192 (16KB) are reasonable lower bounds.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-46（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
