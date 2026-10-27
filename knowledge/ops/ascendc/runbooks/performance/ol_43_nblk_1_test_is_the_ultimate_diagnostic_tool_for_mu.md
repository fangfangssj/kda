---
type: Runbook
title: "nblk=1 test is the ultimate diagnostic tool for multi-core contention"
description: "When precision fails at nblk=56 but passes at nblk=1, the root cause is always a multi-core parallel issue (DataCopy alignment overrun EC-22, buffer overrun, write-write race). Strong diagnostic tool:"
tags: [task.optimization, topic.precision, topic.data_movement.data_copy, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-43-nblk-1-test-is-the-ultimate-diagnostic-tool-for-mu.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-43"]
---

# nblk=1 test is the ultimate diagnostic tool for multi-core contention

> 原 OKF v1 卡号：**OL-43**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - precision test fails at nblk>1

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
When precision fails at nblk=56 but passes at nblk=1, the root cause is always a multi-core parallel issue (DataCopy alignment overrun EC-22, buffer overrun, write-write race). Strong diagnostic tool: change nblk to 1 in pybind11.cpp to confirm.

## 证据
Pad V5 (2026-04-10): nblk=1 → 51/51, nblk=56 → 28/51

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-43（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
