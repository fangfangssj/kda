---
type: Optimization
title: "Pooling multi-strategy priority dispatch + UB shrink loop"
description: "Pooling dispatches multiple tiling strategies by priority; a UB shrink loop cuts the UB budget 4KB per iteration until tile count ≥ core count, trading UB utilization for parallelism."
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.programming_model.execution_model.simt, topic.memory, topic.tiling]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-61-pooling-priority-dispatch-ub-shrink.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Pooling multi-strategy priority dispatch + UB shrink loop

> 原 OKF v1 卡号：**OL-61**。

## 优化点 / 选型

**Trigger**: generating pooling ops (AvgPool / MaxPool, etc.). Loaded by Generator and Analyzer.

CANN registers **multiple tiling strategies** for pooling, dispatched by priority:
- (0) NCHW small-kernel
- (1) NHWC small-kernel
- (2) large-kernel
- (19) SIMT fallback

**Key optimization — UB shrink loop**: when the initial tiling produces fewer tiles than
there are cores, reduce the UB budget by 4KB each iteration, forcing progressively smaller
tiles until `tile_count ≥ core_count`. This deliberately trades UB utilization for parallelism
(idle cores are worse than smaller tiles).

**Second key — channel-size branching in NHWC small-kernel**: when `C × dtype < 64B`
(small channel), use gather instructions to pack multiple spatial positions into one vector
register; when `C × dtype ≥ 64B` (large channel), vectorize along C directly.

**Evidence**: CANN `avg_pool_common_nhwc_small_kernel_tiling.cpp:425-436` (shrink loop),
`:355-371` (gather mode). E1 level (source analysis).
