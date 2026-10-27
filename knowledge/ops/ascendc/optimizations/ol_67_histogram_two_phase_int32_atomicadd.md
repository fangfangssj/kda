---
type: Optimization
title: "Histogram two-phase int32 atomicAdd pattern"
description: "Do not use float atomicAdd for counts (output may read all-zeros before the kernel completes); use phase-1 int32 atomicAdd then a phase-2 kernel to convert dtype, whose launch is the implicit sync point."
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-67-histogram-two-phase-int32-atomicadd.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Histogram two-phase int32 atomicAdd pattern

> 原 OKF v1 卡号：**OL-67**。

## 优化点 / 选型

**Trigger**: histogram / count-class ops (atomicAdd on counters). Loaded by Generator.

Do **not** use float `atomicAdd` for counts. When a float atomicAdd writes directly to the
output buffer, and there is no subsequent kernel launch to act as an implicit sync point, the
output may be read before the kernel completes — yielding all zeros.

**Correct pattern (two-phase):**
1. Phase 1 — int32 `atomicAdd` to accumulate counts.
2. Phase 2 — a separate kernel converts int32 → target dtype. That second kernel launch is
   itself the implicit synchronization point that guarantees Phase 1 has finished.

**Evidence**: Histc case[0] (128 elements): float atomicAdd produced all zeros; the int32
two-phase pattern gave 15/15 PASS. E3 level (measured).
