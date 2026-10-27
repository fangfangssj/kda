---
type: Optimization
title: "Reduction-op selection — WholeReduceMax vs binary fold by data size"
description: "Reduction schemes differ greatly in perf. Official guidance: small data use WholeReduceMax (single instruction, low latency), large data use binary accumulation (BinaryFold mode). Choose ReduceSum/Max/Min impl by data size."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-52-reduction-op-selection-by-data-size.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Reduction-op selection — WholeReduceMax vs binary fold by data size

> 原 OKF v1 卡号：**OL-52**。

## 优化点 / 选型

**场景 / Trigger**: implementing ReduceSum / ReduceMax / ReduceMin.

**选型**: Different reduction schemes have large perf differences. Official guidance — choose based on data size:
- **Small data** → `WholeReduceMax` (single instruction, low latency).
- **Large data** → binary accumulation (**BinaryFold** mode).

The current `BinaryFoldReduceMax` is a reasonable choice but may be simplified in small-data scenarios.

**Source**: hiascend.com best practices (2026-04).
