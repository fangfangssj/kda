---
type: Optimization
title: "SIMD/SIMT mode selection — scatter ops tail-dimension threshold"
description: "Pick SIMT vs SIMD for scatter ops by tail (embedding) dim: scatter_add <128B→SIMT, ≥128B→SIMD; the sort path uses a higher 512B threshold. SIMD is efficient only at ≥1 vector-register width."
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.tiling]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-59-simd-simt-scatter-tail-dim-threshold.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# SIMD/SIMT mode selection — scatter ops tail-dimension threshold

> 原 OKF v1 卡号：**OL-59**。

## 优化点 / 选型

**Trigger**: SIMT vs SIMD decision for scatter-add-class ops. Loaded by Generator and Analyzer.

CANN `scatter_add` selects its execution mode based on the size of the **tail dimension**
(the embedding dim):

- `embDim * dtype_bytes < 128B` → SIMT
- `embDim * dtype_bytes ≥ 128B` → SIMD

The sort path uses a higher threshold: `< 512B` → SIMT sort; `≥ 512B` → SIMD sort.

**Reason**: SIMD vector ops are only efficient when the operand is ≥ 128 bytes — one full
vector-register width. Below that, the per-element SIMT path wins.

**Evidence**: CANN `scatter_add_tiling_base.cpp:199` (`VAR_TAIL_DIM_SIZE = 128`),
`:208-209` (`VAR_TAIL_DIM_SIZE_SORT = 512`). E1 level (source analysis).
