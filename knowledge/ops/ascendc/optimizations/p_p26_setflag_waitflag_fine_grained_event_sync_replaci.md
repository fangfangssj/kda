---
type: Optimization
title: "SetFlag/WaitFlag fine-grained event sync (replacing PipeBarrier<PIPE_ALL>)"
description: "Problem: PipeBarrier<PIPE_ALL>() blocks all pipes, preventing MTE2/VEC/MTE3 parallelism. Correct pattern: use SetFlag/WaitFlag + HardEvent to specify cross-pipe dependencies precisely. cpp // MTE2→Sca"
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p26-setflag-waitflag-fine-grained-event-sync-replaci.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# SetFlag/WaitFlag fine-grained event sync (replacing PipeBarrier<PIPE_ALL>)

> 原 OKF v1 卡号：**P-P26**。

## 优化点

**Problem**: `PipeBarrier<PIPE_ALL>()` blocks all pipes, preventing MTE2/VEC/MTE3 parallelism.

**Correct pattern**: use `SetFlag`/`WaitFlag` + `HardEvent` to specify cross-pipe dependencies precisely.

```cpp
// MTE2→Scalar: can only GetValue from UB after DataCopy completes
event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(id);
WaitFlag<HardEvent::MTE2_S>(id);

// VEC→Scalar: can only GetValue the result after ReduceSum completes
event_t id2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(id2);
WaitFlag<HardEvent::V_S>(id2);

// Scalar→VEC: can only run Muls after SetValue completes
event_t id3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(id3);
WaitFlag<HardEvent::S_V>(id3);

// Scalar→MTE3: can only DataCopyPad back after SetValue completes
event_t id4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
SetFlag<HardEvent::S_MTE3>(id4);
WaitFlag<HardEvent::S_MTE3>(id4);
```

**Interaction with TQue**: TQue<VECIN,2>'s AllocTensor/EnQue/DeQue/FreeTensor manages double-buffering automatically. SetFlag/WaitFlag is for scalar↔vector sync outside of TQue.

**Note**: OL-4 (TQue data corruption) may be a specific buffer-size / configuration issue. E12 expert code uses TQue<VECIN,2> + SetFlag and works correctly.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P26，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: SIMD pipeline overlap；优先级: HIGH。

> 索引行原文：`| P-P26 | SetFlag/WaitFlag fine-grained sync | memory_access | SIMD pipeline overlap | HIGH |`
