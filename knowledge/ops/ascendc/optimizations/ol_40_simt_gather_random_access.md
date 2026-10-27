---
type: Optimization
title: "SIMT gather achieves ~1.0x vs CANN for random-access patterns"
description: "For torch.gather, SIMT per-element (512 threads x 56 blocks) hits mean 0.86x vs CANN by using dcache for random reads instead of per-element DMA (SIMD V1 was 0.006x). Slow on large fp16/bf16 dim=last (0.16-0.4x)."
tags: [task.optimization, topic.programming_model.execution_model.simt]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-40-simt-gather-random-access.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# SIMT gather achieves ~1.0x vs CANN for random-access patterns

> 原 OKF v1 卡号：**OL-40**。

## 优化点 / 选型

**场景 / Trigger**: `torch.gather`, `index_select`, or any per-element indirect-addressing op.

**选型**: For `torch.gather`, SIMT per-element with **512 threads × 56 blocks** achieves mean **0.86x**, median **0.89x** vs CANN.
- SIMT beats CANN on small tensors (~1.2x) and large fp32 (~1.1x).
- SIMT is slower on large fp16/bf16 with `dim=last` (0.16–0.4x).

**Key insight**: SIMT uses **dcache for random reads** instead of per-element DMA (the V1 SIMD approach ran at 0.006x). The remaining gap (esp. the fp16 slow cases) is an optimization knowledge gap, not a hardware limitation — CANN uses the same AscendC API (see OL-42).

**Evidence**: Gather V2 (2026-04-10): SIMD 0.006x → SIMT 0.86x mean (158x improvement).
