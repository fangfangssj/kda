---
type: Runbook
title: "PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S"
description: "On Ascend950PR, pipe_barrier() accepts values [4,6] = PIPE_MTE2, PIPE_V, PIPE_MTE3 only. PIPE_S is NOT valid. For scalar pipe synchronization, use SetFlag/WaitFlag with S_MTE3, S_V, MTE2_S, V_S event"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-35-pipebarrier-only-supports-mte2-v-mte3-not-pipe-s.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-35"]
---

# PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S

> 原 OKF v1 卡号：**OL-35**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when using PipeBarrier in SIMD kernels with scalar operations

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
On Ascend950PR, pipe_barrier() accepts values [4,6] = PIPE_MTE2, PIPE_V, PIPE_MTE3 only. PIPE_S is NOT valid. For scalar pipe synchronization, use SetFlag/WaitFlag with S_MTE3, S_V, MTE2_S, V_S event types.

## 证据
- Sort V1 compile error (2026-04-09), EC-15
  - clipped_swiglu port_a3_to_a5 kw-1 (2026-05-17): scalar-gather of interleaved A/B halves via `SetValue(i, …)` followed by `Cast`/`Mins`/`Maxs` required `HardEvent::S_V`. Replacing the natural `PipeBarrier<PIPE_S>` with `SetFlag/WaitFlag<HardEvent::S_V>` at two sync sites resolved the `kernel_reg.h:85` range error and produced 8/8 cases PASS.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-35（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
