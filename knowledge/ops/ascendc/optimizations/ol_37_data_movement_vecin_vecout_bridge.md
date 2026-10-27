---
type: Optimization
title: "Data-movement ops — use the VECIN→VECOUT bridge pattern"
description: "Pure data-movement kernels need a VEC op between VECIN and VECOUT for pipeline sync; use Adds(dst,src,0.0f) as a no-op bridge (Cast->Adds->Cast for fp16/bf16). Direct UB-to-UB DataCopy is prohibited (PB-9)."
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-37-data-movement-vecin-vecout-bridge.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Data-movement ops — use the VECIN→VECOUT bridge pattern

> 原 OKF v1 卡号：**OL-37**。

## 优化点 / 选型

**场景 / Trigger**: implementing cat, split, pad, repeat, permute, or any pure-copy op.

**Pattern (P-CAT-1)**: Pure data-movement kernels need a VEC op between VECIN (GM→UB) and VECOUT (UB→GM) to maintain pipeline sync. Use `Adds(dst, src, 0.0f, count)` as a no-op bridge. For fp16/bf16, bridge through fp32 via `Cast → Adds → Cast` (lossless roundtrip).

**Constraint / why**: Direct UB-to-UB `DataCopy` is prohibited (PB-9), so a VEC bridge is required rather than a straight copy.

**Evidence**: Cat V2 kernel (2026-04-09), P-CAT-1 pattern. See OL-49 for the more efficient TQueBind alternative to the Adds bridge.
