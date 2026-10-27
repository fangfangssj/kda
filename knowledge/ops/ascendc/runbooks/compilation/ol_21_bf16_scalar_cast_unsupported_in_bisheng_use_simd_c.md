---
type: Runbook
title: "bf16 scalar cast unsupported in bisheng — use SIMD Cast + GetValue"
description: "bisheng compiler (CANN 9.0.0) does not support `static_cast<float>(bfloat16_t)` or reverse. Error: \"not support bf16 type cast\". Half (fp16) scalar cast works fine. The correct approach: use SIMD `Cas"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-21-bf16-scalar-cast-unsupported-in-bisheng-use-simd-c.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-21"]
---

# bf16 scalar cast unsupported in bisheng — use SIMD Cast + GetValue

> 原 OKF v1 卡号：**OL-21**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - bfloat16_t type in kernel code, bf16 GetValue, static_cast<float>(bf16)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
bisheng compiler (CANN 9.0.0) does not support `static_cast<float>(bfloat16_t)` or reverse. Error: "not support bf16 type cast". Half (fp16) scalar cast works fine. The correct approach: use SIMD `Cast(bf16→float)` vector intrinsic on a buffer, then `GetValue(i)` to read float scalar. See P-P27 pattern.

## 证据
tests/repro/bf16_cast_repro.cpp (6 cases), reg_convert.h API inventory

- 相关：[bf16 Scalar Cast Failure](pb_4_bf16_scalar_cast_failure.md)——同一限制的另一种表现形态。

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-21（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
