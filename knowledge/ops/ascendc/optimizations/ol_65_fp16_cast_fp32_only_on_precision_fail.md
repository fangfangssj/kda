---
type: Optimization
title: "fp16 complex math — Cast to fp32 only when precision fails (UPDATED)"
description: "For fp16 complex math (Exp/Div/Erf/Tanh) try native fp16 compute first — on Ascend950PR it is correct and faster (saves 2 Casts); Cast to fp32 only when precision fails. bf16 still needs Cast."
tags: [task.debugging, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-65-fp16-cast-fp32-only-on-precision-fail.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
---

# fp16 complex math — Cast to fp32 only when precision fails (UPDATED)

> 原 OKF v1 卡号：**OL-65**。

## 优化点 / 选型

**Trigger**: fp16 input + complex math (Exp, Div, Reciprocal, Erf, Tanh). Loaded by Generator.

**Lesson (UPDATED 2026-04-15)**: the original conclusion "fp16 Cast to fp32 is faster" was
*overturned* by a GELU re-test. The re-test showed that **native fp16 `Erf()`/`Tanh()` on
Ascend950PR is correct and faster** — it saves 2 Cast ops (~20% improvement on large tensors).
The original 0.65x regression was mainly due to insufficient TQue depth (see OL-63), not to
direct fp16 compute.

**Action**: for fp16 complex math, **try native compute first** by default. Only Cast to fp32
when precision FAILs. bf16 still needs Cast (see PB-4).

**Evidence**: GELU re-test 2026-04-15 — fp16 native `Erf()` PASSes precision and beats the Cast
approach on perf. E3 level (measured).
