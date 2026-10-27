---
type: Runbook
title: "Shared NPU Contention"
description: "Shared NPU Contention"
tags: [task.debugging, topic.data_movement, topic.performance.analysis, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-10-shared-npu-contention.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["PB-10"]
---

# Shared NPU Contention

> 原 OKF v1 卡号：**PB-10**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Benchmark results vary wildly between runs

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Symptom**: Benchmark results vary wildly between runs
- **Affected**: A5 server (shared infrastructure)
- **Workaround**: Run `npu-smi info` before benchmarking, check for other processes
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-15

<!-- 迁移自 porter kb/target/ascendc/（PB-10，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
