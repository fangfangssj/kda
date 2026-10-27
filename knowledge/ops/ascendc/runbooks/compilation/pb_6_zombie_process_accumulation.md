---
type: Runbook
title: "Zombie Process Accumulation"
description: "Zombie Process Accumulation"
tags: [task.debugging, topic.data_movement, topic.performance.analysis, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-6-zombie-process-accumulation.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["PB-6"]
---

# Zombie Process Accumulation

> 原 OKF v1 卡号：**PB-6**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Training/benchmark hangs, resource exhaustion after multiple runs

- **Symptom**: Training/benchmark hangs, resource exhaustion after multiple runs
- **Affected**: Docker containers on A5 server
- **Workaround**: **Always restart container before every experiment**
- **Evidence**: 2280 zombies found after E13h

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Archived

### PB-7 (duplicate, line 68): Shared NPU Contention
- **Archived**: 2026-04-09. Reason: duplicate ID with PB-7 (line 43, merge_mix_obj). Content moved to PB-10.

<!-- 迁移自 porter kb/target/ascendc/（PB-6，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
