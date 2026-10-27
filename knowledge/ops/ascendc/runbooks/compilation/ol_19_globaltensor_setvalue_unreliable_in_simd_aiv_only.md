---
type: Runbook
title: "GlobalTensor::SetValue() unreliable in SIMD AIV-only mode"
description: "`GlobalTensor<float>::SetValue(index, value)` in AIV-only SIMD mode silently drops ~80% of writes. Discovered in E11: SG sorted backward wrote grad_weight[edge_id] via SetValue, 1633/2048 values were"
tags: [task.debugging, topic.data_movement.data_copy, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-19-globaltensor-setvalue-unreliable-in-simd-aiv-only.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-19"]
---

# GlobalTensor::SetValue() unreliable in SIMD AIV-only mode

> 原 OKF v1 卡号：**OL-19**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - writing scalar values to GM in SIMD (KERNEL_TYPE_AIV_ONLY) kernel

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
`GlobalTensor<float>::SetValue(index, value)` in AIV-only SIMD mode silently drops ~80% of writes. Discovered in E11: SG sorted backward wrote grad_weight[edge_id] via SetValue, 1633/2048 values were zero while the kernel logic was proven correct (grad_in via DataCopy was PASS). Root cause unknown — may be CANN 9.0.T501 or bisheng AIV scalar GM write bug.

## 证据
E11 debugging session and on-device ablation

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-19（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
