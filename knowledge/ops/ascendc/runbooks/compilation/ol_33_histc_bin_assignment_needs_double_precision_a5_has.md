---
type: Runbook
title: "Histc bin assignment needs double precision — A5 has no double"
description: "torch.histc uses double precision internally for bin boundary computation. A5 AIV cores only support float32/fp16/bf16. Float kernel gets 1-2 bins wrong at boundaries due to rounding. CPU double fallb"
tags: [task.debugging, topic.precision, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-33-histc-bin-assignment-needs-double-precision-a5-has.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-33"]
---

# Histc bin assignment needs double precision — A5 has no double

> 原 OKF v1 卡号：**OL-33**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when implementing histogram / binning operations

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
torch.histc uses double precision internally for bin boundary computation. A5 AIV cores only support float32/fp16/bf16. Float kernel gets 1-2 bins wrong at boundaries due to rounding. CPU double fallback is NOT acceptable — need float kernel with adjusted bin formula.

## 证据
Histc V1 (2026-04-09), float kernel had 2% mismatch, CPU fallback got 100% match but is cheating

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-33（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
