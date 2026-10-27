---
type: Runbook
title: "GM tiling struct cannot be copied wholesale — must scalar-read each field"
description: "AscendC does not support `auto tmp = *tilingGmPtr` (struct-wide copy from GM to a local variable). It errors as address-space mismatch. **Correct approach**: cast the tiling struct GM pointer to `__gm"
tags: [task.debugging, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-77-gm-tiling-struct-cannot-be-copied-wholesale-must-s.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-77", "__gm__"]
---

# GM tiling struct cannot be copied wholesale — must scalar-read each field

> 原 OKF v1 卡号：**OL-77**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel uses a `__gm__` tiling struct to pass parameters (batch_size, hidden_size, etc. packed into a struct)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
AscendC does not support `auto tmp = *tilingGmPtr` (struct-wide copy from GM to a local variable). It errors as address-space mismatch. **Correct approach**: cast the tiling struct GM pointer to `__gm__ int32_t*` and read field by field: `batchSize_ = *(tilingGmInt + 0); seqLen_ = *(tilingGmInt + 1);`. Or pass each parameter as an independent kernel argument in the pybind layer (avoid struct).

## 证据
30_TimeDecayExponentialStabilization V1: `auto tiling = *tilingData` failed to compile. Switching to scalar reads compiled. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-77（category=platform_compat，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
