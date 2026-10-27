---
type: Optimization
title: "Recompute contiguous strides after leading-one rank padding"
description: "Variable-rank layout and index ops must recompute strides from the fully padded shape; carrying old strides breaks every lower-rank flat-index mapping."
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-285-recompute-strides-after-rank-padding.md
    role: primary
created_at: "2026-07-29T00:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: []
---

# Recompute contiguous strides after leading-one rank padding

> 原 OKF v1 卡号：**OL-285**。

`applies_to: soc=all; cann=all; op_class=variable-rank layout/index op; backend=ascendc`

## Principle

When normalizing variable rank by padding shapes with leading dimensions of size one, do not extend the old stride vector by repeating its first entry. Recompute contiguous strides from the fully padded shape. Otherwise full-rank cases can pass while every lower-rank ravel/unravel mapping is wrong.

```python
def contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return strides

padded_shape = [1] * (max_dims - len(shape)) + list(shape)
padded_strides = contiguous_strides(padded_shape)
```

Test every supported rank and trace a non-zero flat offset through the round trip. Applies to repeat, permute, gather/scatter, pad and backward/sparse-gradient index paths.

**Evidence / provenance**: derived from historical card TR-OL-13. On 16_Repeat (2026-05-17), carrying the old head stride passed rank-4 cases but failed rank-1/2/3; recomputing from the padded shape produced 49/49 PASS. The stride recurrence is mathematical; the failure signature and pass count are measured.
