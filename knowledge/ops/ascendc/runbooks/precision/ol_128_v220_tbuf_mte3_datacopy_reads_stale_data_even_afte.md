---
type: Runbook
title: "V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]"
description: "V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-128-v220-tbuf-mte3-datacopy-reads-stale-data-even-afte.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["OL-128"]
---

# V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]

> 原 OKF v1 卡号：**OL-128**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - paradigm: ascendc

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

**Principle**: On V220 (CANN 9.0.0), when VEC ops write to a `TBuf<VECCALC>` and a subsequent `DataCopy` (MTE3) reads from the same or a different TBuf, the MTE3 read may return **stale data from a different buffer region** — even when `SetFlag<HardEvent::V_MTE3>/WaitFlag` fences are in place.

**Fix**: Use `TQue<QuePosition::VECOUT, depth=1>` for output instead of bare `TBuf<VECCALC>` + `DataCopy`. The TQue's `EnQue/DeQue` rotation provides hardware-managed V→MTE3 sync.

**Evidence**: op#31 IOU DS kw-3 (2026-05-01): 5 precision iterations, DataCopy always read wrong buffer. kw-4 applied TQue<VECOUT> → MTE3 coherence fixed.

**Other instances (predicted)**: any V220 kernel where TBuf is used for output staging before DataCopy to GM.

**Cross-ref**: PB-22 (MTE2 DataCopy 32-byte limit), OL-127 (CANN API surface gaps V220), OL-94 (TQue vs TBuf decision).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-128（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
