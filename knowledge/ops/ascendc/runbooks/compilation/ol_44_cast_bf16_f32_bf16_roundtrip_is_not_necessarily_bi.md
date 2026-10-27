---
type: Runbook
title: "Cast(bf16→f32→bf16) roundtrip is not necessarily bit-exact"
description: "Cast(bf16→f32, CAST_NONE) is lossless (bf16 is a subset of f32), but Cast(f32→bf16, CAST_ROUND) may alter f32 bit patterns through intermediate VEC ops (such as an Adds bridge), producing a different"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-44-cast-bf16-f32-bf16-roundtrip-is-not-necessarily-bi.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-44"]
---

# Cast(bf16→f32→bf16) roundtrip is not necessarily bit-exact

> 原 OKF v1 卡号：**OL-44**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - bf16 data-movement kernel exhibits precision mismatch

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Cast(bf16→f32, CAST_NONE) is lossless (bf16 is a subset of f32), but Cast(f32→bf16, CAST_ROUND) may alter f32 bit patterns through intermediate VEC ops (such as an Adds bridge), producing a different bf16 after the roundtrip. In pure-copy kernels, use CAST_NONE for the reverse conversion (truncate to recover original bf16 bits), or SIMT direct assignment to bypass Cast.

## 证据
Pad V4 (2026-04-10): bf16 Cast roundtrip regression

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-44（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
