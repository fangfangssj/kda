---
type: Optimization
title: "`TQueBind<VECIN, VECOUT, 1>` for in-place UB buffer reuse"
description: "Trigger: a compute pass reads a buffer, mutates in-place, writes the same buffer back to GM. Separate TQue<VECIN> + TQue<VECOUT> would consume 2× the UB. Pattern: cpp TQueBind<QuePosition::VECIN, QueP"
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p66-tquebind-vecin-vecout-1-for-in-place-ub-buffer-r.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: []
---

# `TQueBind<VECIN, VECOUT, 1>` for in-place UB buffer reuse

> 原 OKF v1 卡号：**P-P66**。

## 优化点

**Trigger**: a compute pass reads a buffer, mutates in-place, writes the same buffer back to GM. Separate `TQue<VECIN>` + `TQue<VECOUT>` would consume 2× the UB.

**Pattern**:
```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer1;

pipe_->InitBuffer(buffer1, /*depth=*/1, /*bytes=*/tileSize);

// CopyIn:
LocalTensor<T> local = buffer1.AllocTensor<T>();
DataCopyPad(local, srcGm[offset], params, padParams);
buffer1.EnQue(local);

// Compute (mutate in place):
LocalTensor<T> local = buffer1.DeQue<T>();
Cast(local, local, RoundMode::CAST_NONE, count);
Adds(local, local, 1, count);
buffer1.EnQue(local);

// CopyOut:
LocalTensor<T> local = buffer1.DeQue<T>();
DataCopyPad(dstGm[offset], local, params);
```

**Trade-off**:
- `TQueBind<VECIN, VECOUT, 1>` = 1 × tileSize UB, forced serial CopyIn → Compute → CopyOut on each tile.
- `TQue<VECIN, depth=2>` + `TQue<VECOUT, depth=2>` = 4 × tileSize UB, but CopyIn(iter+1) overlaps Compute(iter).

**When to use**:
- UB budget is tight (multi-tensor fused ops where every KB matters; cf. op#11 DequantSwigluQuant peak 194/192 KB).
- Op is memory-bound on small tiles (pipeline depth doesn't help — bandwidth saturates anyway).
- In-place compute fits in one VEC sequence (no need to keep src + dst alive simultaneously).

**When NOT to use**:
- Op is VEC-bound and benefits from depth ≥ 2 pipelining (don't trade throughput for UB).
- Input and output dtypes/sizes differ (would need 2 separate physical slots regardless).

**Related to PB-17 (P-P65)**: PB-17 is the aggressive form — manually aliasing two named TBufs onto the same physical UB slot, which the framework does not manage. `TQueBind` is the framework-managed version with proper EnQue/DeQue protocol; **use it first**, only escalate to manual aliasing if `TQueBind` doesn't fit the access pattern.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P66，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

同号异题提示：私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）的 P-P66 与 v1 卡同号不同题；该来源的 P-P66 已另立新卡 [runbooks/runtime/pybind_h_axis_alignment_padding_convention.md](../runbooks/runtime/pybind_h_axis_alignment_padding_convention.md)，引用 P-P66 时需按主题区分。
