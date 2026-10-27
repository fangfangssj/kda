---
type: Runbook
title: "int64 index tensor in SIMT — direct __gm__ read works"
description: "Reading int64_t from GM via `__gm__ int64_t* idx; int64_t val = idx[i];` works correctly in SIMT. No special handling needed for 8-byte types. This avoids the int64→int32 conversion overhead in pybind"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-41-int64-index-tensor-in-simt-direct-gm-read-works.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-41"]
---

# int64 index tensor in SIMT — direct __gm__ read works

> 原 OKF v1 卡号：**OL-41**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernels reading int64 values from GM in SIMT mode

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Reading int64_t from GM via `__gm__ int64_t* idx; int64_t val = idx[i];` works correctly in SIMT. No special handling needed for 8-byte types. This avoids the int64→int32 conversion overhead in pybind11.

## 证据
Gather V2 (2026-04-10): 47/47 PASS with direct int64 index reads

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-41（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
