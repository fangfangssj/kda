---
type: Runbook
title: "SIMT 1024 threads crashes with 507035 on Ascend950PR"
description: "Raising LAUNCH_BOUND from 512 to 1024 caused a 507035 vector core exception. Not every kernel can run 1024 threads — kernel complexity (int64 division loops etc.) may exceed register-file capacity. Sa"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-56-simt-1024-threads-crashes-with-507035-on-ascend950.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-56"]
---

# SIMT 1024 threads crashes with 507035 on Ascend950PR

> 原 OKF v1 卡号：**OL-56**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMT kernel attempts to raise thread count to 1024

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Raising LAUNCH_BOUND from 512 to 1024 caused a 507035 vector core exception. Not every kernel can run 1024 threads — kernel complexity (int64 division loops etc.) may exceed register-file capacity. Safe practice: compute-heavy SIMT kernels keep 512 threads.

## 证据
Permute kernel int64 division loop + 1024 threads → 507035 crash

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-56（category=platform_constraint，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
