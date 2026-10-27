---
type: Runbook
title: "View-returning ops have ~0 reference cost — benchmark vs materialized copy"
description: "torch.split/narrow/slice return tensor views (~0.009ms), not copies. Our AscendC kernel does actual data movement. Direct latency comparison is misleading — our kernel is \"slower\" but does real work."
tags: [task.optimization, topic.data_movement, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-39-view-returning-ops-have-0-reference-cost-benchmark.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-39"]
---

# View-returning ops have ~0 reference cost — benchmark vs materialized copy

> 原 OKF v1 卡号：**OL-39**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - when benchmarking split, narrow, slice, or any op that returns views

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
torch.split/narrow/slice return tensor views (~0.009ms), not copies. Our AscendC kernel does actual data movement. Direct latency comparison is misleading — our kernel is "slower" but does real work. For fair evaluation, compare against the cost that would force materialization (.contiguous() on the view).

## 证据
Split 14 benchmark (2026-04-09): torch.split ~0.009ms vs AscendC 0.018-0.96ms

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-39（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
