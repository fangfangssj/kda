---
type: Optimization
title: "Verify nested outputs structurally and require every leaf to pass"
description: "Compare container type and arity before recursively checking every tensor leaf; multi-gradient backward passes only when every output passes."
tags: [task.optimization, topic.implementation.masking]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-289-verify-nested-outputs-by-leaf.md
    role: primary
created_at: "2026-07-29T00:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: []
---

# Verify nested outputs structurally and require every leaf to pass

> 原 OKF v1 卡号：**OL-289**。

`applies_to: soc=all; cann=all; op_class=verifier/backward truth; backend=ascendc`

## Principle

A verifier for tensor-or-tuple/list outputs must compare container kind, arity and recursively each leaf. Never let `zip` silently truncate mismatched outputs. The aggregate passes only when every leaf passes; report the failing path and the worst finite metric without hiding a structural failure.

```python
def compare_tree(actual, expected, path="output"):
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        return [compare_tree(a, e, f"{path}[{i}]")
                for i, (a, e) in enumerate(zip(actual, expected))]
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    return compare_tensor(actual, expected, path)
```

This is mandatory for multi-gradient backward ops: a correct `dX` must not mask a missing or bad `dWeight`/`dBias`.

**Evidence / provenance**: derived from historical card TR-OL-26. On 22_Nonzero (2026-05-17), a tensor-or-tuple wrapper covered alternating `as_tuple` cases and the full 50/50 suite passed. The explicit arity guard is a strengthened correctness deduction that prevents `zip` truncation.
