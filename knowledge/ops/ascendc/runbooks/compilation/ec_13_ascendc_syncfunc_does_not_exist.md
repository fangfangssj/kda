---
type: Runbook
title: "`AscendC::SyncFunc<>` does not exist"
description: "`AscendC::SyncFunc<>` does not exist"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-13-ascendc-syncfunc-does-not-exist.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-13", "AscendC::SyncFunc<>"]
---

# `AscendC::SyncFunc<>` does not exist

> 原 OKF v1 卡号：**EC-13**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: no member named 'SyncFunc' in namespace 'AscendC'
  ```
- **Root cause**: There is no `AscendC::SyncFunc` API. The generated code (from templates or LLM) may invent this API for pipe synchronization. The correct API uses `SetFlag`/`WaitFlag` with event IDs fetched from `GetTPipePtr()->FetchEventID()`.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();

  // AFTER (compiles):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ev);
  ```
- **Common sync events**: MTE2_S (GM→scalar), S_MTE3 (scalar→GM write), V_S (VEC→scalar), S_V (scalar→VEC)
- **Evidence**: Cumsum V1 build failure (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-13，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
