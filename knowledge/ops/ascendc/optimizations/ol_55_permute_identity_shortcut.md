---
type: Optimization
title: "Permute — identity shortcut is the largest optimization lever"
description: "Detect identity permutations in the pybind layer and return the input directly instead of cloning (Permute median rose 0.04x→0.63x). Data-movement ops should first check whether data actually needs to move."
tags: [task.optimization, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-55-permute-identity-shortcut.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Permute — identity shortcut is the largest optimization lever

> 原 OKF v1 卡号：**OL-55**。

## 优化点 / 选型

**Trigger**: `torch.permute` / transpose-style ops. Loaded by Generator and Optimizer.

Many benchmarks include *identity* permutations where `perm = [0, 1, ..., ndim-1]`. For
these the reference `torch.permute(x, identity).contiguous()` is a no-op — it returns the
original tensor unchanged. If the kernel unconditionally does a clone/copy on identity cases,
its perf is 100x+ slower than the reference, which never moves data at all.

**Fix**: detect the identity perm in the pybind layer and directly return the input tensor,
skipping the kernel entirely. In the Permute benchmark, 17 of 149 cases are identity; after
adding this shortcut the overall median improved from 0.04x (V1) → 0.63x (V2).

**Generalization**: any data-movement op should first ask "does the data actually need to
move?" and short-circuit the trivial case before dispatching a copy kernel.
