---
type: Runbook
title: "Shared NPU — performance data affected by concurrent users"
description: "Shared NPU — performance data affected by concurrent users"
tags: [task.optimization, topic.data_movement, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-15-shared-npu-performance-data-affected-by-concurrent.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-15"]
---

# Shared NPU — performance data affected by concurrent users

> 原 OKF v1 卡号：**OL-15**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - when running performance benchmarks on A5 server

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: environment
- **Loaded by**: QA
- **Trigger**: when running performance benchmarks on A5 server
- **Lesson**: A5 server is shared. Our container binds one NPU but other processes may use it. Before performance tests: check `npu-smi info` for other processes. If busy: try another NPU or wait/retry. Never trust a single benchmark run on shared infra.
- **Evidence**: A5 server 198.51.100.35 shared infrastructure

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-15（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
