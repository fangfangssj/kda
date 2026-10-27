---
type: Runbook
title: "Restart container before experiments (zombie processes)"
description: "Restart container before experiments (zombie processes)"
tags: [task.debugging, topic.data_movement, topic.memory, topic.performance.analysis, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-6-restart-container-before-experiments-zombie-proces.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-6"]
---

# Restart container before experiments (zombie processes)

> 原 OKF v1 卡号：**OL-6**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - before launching any benchmark or experiment on the A5 container

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: environment
- **Loaded by**: QA
- **Trigger**: before launching any benchmark or experiment on the A5 container
- **Lesson**: After long experiment sessions, zombie processes accumulate from previous NPU kernel launches. 2280 zombie processes were found after one extended run. These cause training hangs, kernel launch failures, resource exhaustion, and unreproducible timing results. Always restart the container (`docker restart can_torch_cann_device_1`) before every experiment session.
- **Evidence**: CLAUDE.md (global) "CRITICAL: ALWAYS restart container before EVERY experiment", MEMORY.md

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-6（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
