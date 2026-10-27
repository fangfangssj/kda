---
type: Optimization
title: "N-dim concat decomposes to 3D (outer × cat_dim × inner)"
description: "Decompose cat(tensors, dim=d) into flat 3D: outer=prod(shape[:d]), cat_dim=shape[d], inner=prod(shape[d+1:]). outer==1 is contiguous copy; outer>1 is per-outer strided copy; one kernel per input tensor."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-38-ndim-concat-decompose-3d.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# N-dim concat decomposes to 3D (outer × cat_dim × inner)

> 原 OKF v1 卡号：**OL-38**。

## 优化点 / 选型

**场景 / Trigger**: implementing `torch.cat` along an arbitrary dim.

**选型 / Decomposition**: Decompose `cat(tensors, dim=d)` into a flat 3D shape:
- `outer = prod(shape[:d])`
- `cat_dim = shape[d]`
- `inner = prod(shape[d+1:])`

**Cases**:
- `outer == 1` → flat contiguous copy.
- `outer > 1` → per-outer chunked copy with stride.
- Launch one kernel per input tensor.
- Non-aligned chunks need an overlapping tail write (EC-16).

**Evidence**: Cat V2 kernel (2026-04-09). The same 3D decomposition is applicable to Split and Permute.
