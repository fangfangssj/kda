---
type: Runbook
title: "Cast-to-Int32 workspace for non-float atomicAdd"
description: "CANN's scatter_add uses a three-phase scheme for fp16/bf16/int8: Phase 1 cast output tensor to int32 into a workspace; Phase 2 atomicAdd on the int32 workspace; Phase 3 cast back from workspace to the"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.atomic, topic.memory, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-58-cast-to-int32-workspace-for-non-float-atomicadd.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-58"]
---

# Cast-to-Int32 workspace for non-float atomicAdd

> 原 OKF v1 卡号：**OL-58**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - scatter-add kernel uses fp16/bf16/int8 or other non-native atomicAdd types

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
CANN's scatter_add uses a three-phase scheme for fp16/bf16/int8: Phase 1 cast output tensor to int32 into a workspace; Phase 2 atomicAdd on the int32 workspace; Phase 3 cast back from workspace to the original dtype. int32 atomicAdd is natively supported on all Ascend hardware. Between phases, use SyncAll() for full-core sync + pipe_.Reset() to reclaim UB.

## 证据
CANN scatter_add_simd.h:365-389, scatter_add_common.h:72-93. E1 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-58（category=platform_constraint，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
