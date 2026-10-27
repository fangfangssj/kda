---
type: Runbook
title: "SIMD multi-core GM scalar write has 8-element granularity contention"
description: "On A5, the minimum granularity of `DataCopy(GM, UB, count)` is 32B (fp32: 8 elements; fp16/bf16: 16 elements). If multiple cores write adjacent positions in a GM array (uid → array[uid]), DataCopy act"
tags: [task.debugging, topic.data_movement.data_copy, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-70-simd-multi-core-gm-scalar-write-has-8-element-gran.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-70"]
---

# SIMD multi-core GM scalar write has 8-element granularity contention

> 原 OKF v1 卡号：**OL-70**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMD kernel where multiple AIV cores each write a scalar (mean/rstd/sum etc.) to a shared GM array

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
On A5, the minimum granularity of `DataCopy(GM, UB, count)` is 32B (fp32: 8 elements; fp16/bf16: 16 elements). If multiple cores write adjacent positions in a GM array (uid → array[uid]), DataCopy actually writes that 8/16-element block, which **overwrites neighboring cores' data** → non-deterministic / all-zero output. **Fix (see P-P49)**: allocate an isolated 8-aligned slot per uid (uid * 8), and have the host (pybind) extract the real value via stride select.

## 证据
2_GroupNormSwish: mean/rstd is one fp32 scalar per (N, group). Direct `out[uid] = value` then DataCopy cross-core overwrote each other intermittently. Changed to `out[uid * 8] = value` + pybind extraction → PASS. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-70（category=platform_compat，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
