---
type: Runbook
title: "Avoid UB bank conflicts — consecutive VEC instructions should hit different banks"
description: "UB is composed of multiple banks. When multiple VEC instructions concurrently read/write different addresses in the same bank, a bank conflict occurs and instructions must queue. Solution: ensure the"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.memory, topic.performance.analysis, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-50-avoid-ub-bank-conflicts-consecutive-vec-instructio.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-50"]
---

# Avoid UB bank conflicts — consecutive VEC instructions should hit different banks

> 原 OKF v1 卡号：**OL-50**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMD kernel perf is below expectation and msprof shows low VEC pipe utilization

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
UB is composed of multiple banks. When multiple VEC instructions concurrently read/write different addresses in the same bank, a bank conflict occurs and instructions must queue. Solution: ensure the source and destination tensors of consecutive VEC ops are in different banks (adjust buffer allocation offsets). UB bank count and size vary by chip version.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-50（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
