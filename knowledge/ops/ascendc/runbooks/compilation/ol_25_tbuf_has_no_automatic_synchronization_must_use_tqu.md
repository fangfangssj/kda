---
type: Runbook
title: "TBuf has NO automatic synchronization — must use TQue or explicit sync"
description: "TBuf has NO automatic synchronization — must use TQue or explicit sync"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement, topic.memory, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-25-tbuf-has-no-automatic-synchronization-must-use-tqu.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-25"]
---

# TBuf has NO automatic synchronization — must use TQue or explicit sync

> 原 OKF v1 卡号：**OL-25**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - mixing TBuf with TQue in the same kernel, or using TBuf for accumulators alongside DMA operations

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: platform_bug (actually: design feature)
- **Loaded by**: Builder, Optimizer
- **Trigger**: mixing TBuf with TQue in the same kernel, or using TBuf for accumulators alongside DMA operations
- **Lesson**: E14 TQue refactoring of backward sorted kernel FAILED (max_diff=0.76) because accum was in TBuf while input was in TQue. Root cause: TBuf.Get() has **no EnQue/DeQue → no hardware set/wait signals**. When VEC writes to TBuf accum and MTE2 writes to TQue input concurrently, there's no sync → UB bus contention → data corruption.
  Fix: move accum to TQue<VECOUT> (same pattern as working forward PingPong). Official docs confirm: "TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作" and "EnQue调用会发射同步指令set" — TBuf simply has no sync mechanism.
- **Rule**: When using TQue for DMA, ALL VEC-accessed buffers must EITHER be TQue-managed OR have explicit SyncFunc/SetFlag/WaitFlag. TBuf + TQue without explicit sync = data corruption.
- **Evidence**: E14 revert (2026-04-06), CANN 9.0.0-beta.2 official docs "编程模型设计原理", ASCENDC_LANGUAGE_REFERENCE.md

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-25（category=platform_bug (actually: design feature)，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
