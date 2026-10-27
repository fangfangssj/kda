---
type: Optimization
title: "SIMT is a general solution to EC-22 (DataCopy alignment overrun)"
description: "When SIMD DataCopy alignment overrun (EC-22) is hard to fix, switch to SIMT as correctness-first: one element per thread avoids DataCopy/alignment entirely. Large tensors slower (0.04-0.05x), small/medium faster (1.2-2.8x)."
tags: [task.optimization, topic.precision, topic.programming_model.execution_model.simt, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-45-simt-solves-datacopy-alignment-overrun.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# SIMT is a general solution to EC-22 (DataCopy alignment overrun)

> 原 OKF v1 卡号：**OL-45**。

## 优化点 / 选型

**场景 / Trigger**: a multi-core SIMD kernel hits precision issues due to DataCopy alignment overrun (EC-22).

**选型**: When SIMD DataCopy alignment overrun is hard to fix (many tile types × many dtypes × row-tail cross-block overrun), switch to **SIMT** as the correctness-first solution. SIMT writes one element per thread, completely avoiding DataCopy and alignment issues.

**Tradeoff (perf)**:
- Large tensors slower (SIMT 0.04–0.05x vs SIMD potentially >1.0x).
- Small/medium tensors and non-constant mode can be faster (SIMT 1.2–2.8x).
- Overall mean improved 0.05x → 0.72x.

**Evidence**: Pad V4 (2026-04-10): SIMD V3 28/51 PASS → SIMT V4 51/51 PASS, mean 0.05x → 0.72x.
