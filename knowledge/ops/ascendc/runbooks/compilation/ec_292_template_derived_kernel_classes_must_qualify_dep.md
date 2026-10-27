---
type: Runbook
title: "template-derived kernel classes must qualify dependent-base members with `this->` — bare `pipe_.InitBuffer(...)` fails name lookup (C++ two-phase lookup)"
description: "模板类派生自模板基类时，基类成员（如 `pipe_`）要写 `this->pipe_` 才解析。"
tags: [task.debugging, topic.implementation.synchronization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-292", pipe_]
---

# template-derived kernel classes must qualify dependent-base members with `this->` — bare `pipe_.InitBuffer(...)` fails name lookup (C++ two-phase lookup)

`applies_to: soc=all (C++ language rule, observed on the Ascend310P3 lane); cann=9.1.0; bisheng=(310p lane); op_class=all (kernels using a templated base class holding TPipe/TQue/TBuf members)`

- **Symptom**: compile error of the "use of undeclared identifier 'pipe_'" class inside a template class deriving from a templated base, on a line like `pipe_.InitBuffer(...)` where `pipe_` is a member of the dependent base.
- **Root cause**: C++ two-phase name lookup — members of a dependent base class are not found by unqualified lookup at template definition time.
- **Fix**: qualify the member: `this->pipe_.InitBuffer(...)` (or `Base<T>::pipe_`).
- **Evidence**: instance_norm_grad 310p (2026-09-02, benchmark-acceptance lane, failures_ledger build iter 1 `instance_norm_grad/failures_ledger.md`): bare `pipe_.InitBuffer` in a template-derived class rejected; `this->pipe_` fixed it in 1 compile iteration.
- **Other instances (predicted)**: any kernel refactor that introduces an inheritance layer (a shared base holding `TPipe pipe_`, tiling structs, or queues) over a template parameter. backend=ascendc.
