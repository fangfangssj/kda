---
type: Optimization
title: "SIMD PipeBarrier and DataCopy alignment"
description: "Anti-pattern: PipeBarrier<PIPE_MTE2>() — fine-grained barrier can cause data races. Correct pattern: PipeBarrier<PIPE_ALL>() guarantees correctness. Optimal: use TQue depth=2 double buffering. DataCop"
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_model.simt, topic.data_movement.data_copy, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/f-p4-simd-pipebarrier-and-datacopy-alignment.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# SIMD PipeBarrier and DataCopy alignment

> 原 OKF v1 卡号：**F-P4**。

## 优化点

**Anti-pattern**: `PipeBarrier<PIPE_MTE2>()` — fine-grained barrier can cause data races.

**Correct pattern**: `PipeBarrier<PIPE_ALL>()` guarantees correctness. Optimal: use TQue depth=2 double buffering.

**DataCopy alignment requirement**: fp32: %8==0, fp16/bf16: %16==0. When unaligned, fall back to SIMT or use `DataCopyPad`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（F-P4，convert_patterns_to_okf.py）。confidence 未升格。 -->
