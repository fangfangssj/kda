---
type: Optimization
title: "Slice-D workspace — general splitting scheme for large-hidden-dim ops"
description: "When hidden dim D is too large to fit a row in UB (D×4×2>UB), split the row into UB-sized slices with a GM workspace and accumulate scalar reductions (sum/max) in registers across slices."
tags: [task.optimization, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-60-slice-d-workspace-large-hidden-dim.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Slice-D workspace — general splitting scheme for large-hidden-dim ops

> 原 OKF v1 卡号：**OL-60**。

## 优化点 / 选型

**Trigger**: hidden dimension D too large to fit an entire row in UB — norm, quant, attention,
etc. Loaded by Generator and Optimizer.

When `D × sizeof(float) × 2 > UB`, a whole row cannot be held in UB. General scheme: split the
row into UB-sized slices and use a GM workspace for intermediate results. **Key**: scalar
reduction values (sum, max) are accumulated in *registers* across slices, so they survive the
slice loop without a workspace round-trip.

Concrete 3-pass pattern (from `add_rms_norm_dynamic_quant_cut_d`):
1. slice-by-slice sum-of-squares → register-accumulate → compute `rstd`
2. slice-by-slice `norm × smooth` → write to workspace → track running max in a register
3. slice-by-slice read workspace → `scale + quantize` → write output

**Evidence**: CANN `add_rms_norm_dynamic_quant_cut_d_kernel.h:60-125`. E1 level (source analysis).
