---
type: Runbook
title: "V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]"
description: "V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.pipeline, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-24-v220-tbuf-getvalue-returns-sequential-position-val.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["PB-24", "GetValue", "CopyTile"]
---

# V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]

> 原 OKF v1 卡号：**PB-24**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - When scalar values are read from a TBuf<VECCALC> via GetValue() AND the same kernel also uses TQue EnQue/DeQue (CopyTile) in interleaved fashion, the TBuf GetVa

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Severity**: CRITICAL (silent data corruption — kernel produces wrong output with no error)
- **Status**: CONFIRMED 2026-05-02 op#18 Index pp-1 (12 bisection iterations, V220 CANN 9.0.0)
- **Symptom**: When scalar values are read from a `TBuf<VECCALC>` via `GetValue()` AND the same kernel also uses TQue `EnQue/DeQue` (CopyTile) in interleaved fashion, the TBuf GetValue returns the **sequential position index** (0, 1, 2, ...) instead of the actual stored value. All GM scalar read paths are affected: `GlobalTensor::GetValue`, raw `__gm__` pointer dereference, TBuf `DataCopy` with MTE2_V sync.
- **Fix**: Read ALL TBuf values into a local C++ stack array BEFORE any TQue CopyTile operations begin. This eliminates the interleaving that triggers the corruption.
  ```cpp
  // Read all indices before any TQue operations
  int32_t idx_buf[MAX_INDICES];
  for (int i = 0; i < n_indices; ++i)
      idx_buf[i] = idxBuf_.GetValue<int32_t>(i);
  // NOW start TQue CopyTile operations using idx_buf[]
  ```
- **Evidence**: op#18 Index DS kw-1 (2026-05-02): 9 iterations tested all GM read paths — all returned sequential positions. pp-1 (2026-05-02): 12 bisection iterations confirmed the interleaving root cause. kw-2 applied the "read-all-before-TQue" fix → 41/41 PASS (bit-exact, MERE=0, MARE=0), perf 10.78x vs CANN.
- **Other instances (predicted)**: any V220 kernel that mixes TBuf scalar reads with TQue pipeline operations, especially index/gather/scatter ops where indices are loaded via TBuf.
- **Cross-ref**: PB-22 (MTE2 DataCopy 32B limit), OL-124 (TBuf→MTE3 coherence), OL-123 (V220 API gaps).

<!-- 迁移自 porter kb/target/ascendc/（PB-24，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
