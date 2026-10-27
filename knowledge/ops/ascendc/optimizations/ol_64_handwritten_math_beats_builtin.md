---
type: Optimization
title: "Hand-written math approximations can beat built-in functions"
description: "Hand-written polynomial approximations (e.g. A&S erf for GELU) can beat built-in Erf/Tanh/Sigmoid because they map to pure VEC ops; try built-ins first for precision, hand-roll only if perf is missed."
tags: [task.optimization, topic.precision, topic.implementation.pipeline, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-64-handwritten-math-beats-builtin.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Hand-written math approximations can beat built-in functions

> 原 OKF v1 卡号：**OL-64**。

## 优化点 / 选型

**Trigger**: using built-in math such as `Erf()`, `Tanh()`, `Sigmoid()`. Loaded by Generator
and Optimizer.

The old GELU used the Abramowitz & Stegun erf polynomial approximation (~20 VEC instructions:
Muls, Adds, Mul, Exp, Reciprocal, Div); the new version switched to the built-in `Erf()`. The
old, hand-written version is *faster*.

**Hypothesized reason**: the hand-written polynomial maps entirely to pure VEC instructions
that run continuously in the hardware pipeline, whereas built-in transcendental functions may
use table lookup + interpolation, giving different (worse, here) pipeline efficiency.

**Action**: do not blindly assume built-ins are always fastest. For perf-sensitive ops,
consider a hand-written polynomial approximation. But prefer built-ins **first** for precision
assurance, and switch to a hand-written version only when perf is not met.

**Evidence**: GELU regression 2026-04-14. E1 level (source-analysis inference; exact perf delta
not isolated).
