---
type: Runbook
title: "`using namespace AscendC::Simt;` causes GetBlockIdx ambiguity"
description: "CANN has TWO `GetBlockIdx()` — `AscendC::Simt::GetBlockIdx()` (int32_t) and basic_api `GetBlockIdx()` (int64_t). Using `using namespace AscendC::Simt;` brings both into scope → compile error \"call to"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-14-using-namespace-ascendc-simt-causes-getblockidx-am.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-14"]
---

# `using namespace AscendC::Simt;` causes GetBlockIdx ambiguity

> 原 OKF v1 卡号：**OL-14**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when writing AscendC SIMT kernel headers

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
CANN has TWO `GetBlockIdx()` — `AscendC::Simt::GetBlockIdx()` (int32_t) and basic_api `GetBlockIdx()` (int64_t). Using `using namespace AscendC::Simt;` brings both into scope → compile error "call to GetBlockIdx is ambiguous". Fix: use ONLY `using namespace AscendC;` (without `::Simt`). Dispatchers already use `Simt::VF_CALL` with qualified prefix.

## 证据
SG forward generated kernel, 20 ambiguity errors. Pooling (only `using namespace AscendC;`) compiled fine.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-14（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
