---
type: Runbook
title: "After optimizing one kernel, must audit all similar kernels"
description: "After optimizing one kernel, must audit all similar kernels"
tags: [task.optimization, topic.implementation.synchronization, topic.data_movement, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-24-after-optimizing-one-kernel-must-audit-all-similar.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-24"]
---

# After optimizing one kernel, must audit all similar kernels

> 原 OKF v1 卡号：**OL-24**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - any optimization applied to one kernel variant (forward/backward, fp32/fp16/bf16)

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, Optimizer, Lead
- **Trigger**: any optimization applied to one kernel variant (forward/backward, fp32/fp16/bf16)
- **Lesson**: E13 switching Forward PipeBarrier→TQue gave 1.6-2.3x speedup, but we did not check whether Backward Sorted had the same PipeBarrier anti-pattern (it did — 7 PIPE_ALL). Only discovered when the expert pointed it out at E14.
  After applying an optimization, must:
  1. grep all kernel files for similar anti-patterns
  2. Evaluate whether the same optimization applies to each match
  3. Distinguish "production path" from "non-production path" — only optimize the production path
- **Evidence**: E14 feedback (2026-04-06), sparse_gather_simd.h BackwardSimdSortedF32 had 7 unoptimized PIPE_ALL

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-24（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
