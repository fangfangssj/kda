---
type: Runbook
title: "DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops"
description: "DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-11-datacopy-to-tbuf-veccalc-silent-corruption-on-mult.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["PB-11"]
---

# DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops

> 原 OKF v1 卡号：**PB-11**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count) with manual SetFlag<MTE2_V>/WaitFlag<MTE2_V> sync produces correct data on the first loop iteration but stal

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (17_EmbeddingWithInitialLayernormBackward). Do not downgrade.
- **Symptom**: `DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count)` with manual `SetFlag<MTE2_V>/WaitFlag<MTE2_V>` sync produces correct data on the first loop iteration but stale/corrupt data on subsequent iterations. Related to PB-9 (both are DataCopy corruption) but distinct mechanism: PB-9 is UB→UB; PB-11 is GM→VECCALC with manual sync in a loop.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use `TQue<VECIN, 1>` instead of `TBuf<VECCALC>` for any buffer that receives DataCopy from GM in a loop. The TQue's `AllocTensor/EnQue/DeQue/FreeTensor` pattern provides reliable MTE2→VEC synchronization. Single-iteration usage of TBuf<VECCALC> with DataCopy appears safe.
- **Status**: OPEN
- **Evidence**: DynamicQuant (#29) smooth_scales — 3 cases with row_size > TILE_SIZE failed (2.7%-22.6% mismatch, max_abs_diff=252) due to stale smooth_scales data on 2nd+ tile. First mismatch always at exact TILE_SIZE boundary. Fixed by switching to TQue<VECIN,1>. 42/42 PASS after fix.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## How to Add New Bugs

Append to the appropriate section with:
```
### PB-N: Short Description
- **Symptom**: What you observe
- **Affected**: Platform/version
- **Workaround**: How to work around it
- **Status**: OPEN/FIXED(version)/BY_DESIGN
- **Evidence**: Link to test/doc
```

<!-- 迁移自 porter kb/target/ascendc/（PB-11，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
