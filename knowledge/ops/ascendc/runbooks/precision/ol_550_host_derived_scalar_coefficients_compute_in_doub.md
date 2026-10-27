---
type: Runbook
title: "host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype"
description: "host 侧推导标量系数（如 AdamW 偏置校正）要用 double 算、单次转 fp32——torch wrapped-scalar 语义就是 fp64→fp32 单转，fp32 链式推导差 1 ULP"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.scalar, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-550"]
---

# host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype

**OL-550** — **host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype** — torch wrapped-scalar semantics feed float64→fp32-cast scalars, so fp32-step-derived chains are 1 ULP off (`float32(1.0-0.9) != float32(1.0)-float32(0.9)`). Binds optimizer bias corrections, momentum decays, LR schedules, 1−momentum factors — any Python-level formula applied to fp32 tensors. 17_AdamW 310p 2026-08-15: load-bearing for m/v bit-exactness (MERE 0.0). Different level from P-P51 (in-kernel chain dtype — both can bind the same op at different sites). backend=all.

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (host/pybind-side numerics)`
<!-- applies_to_backend: all -->

**Principle**: when a PyTorch reference combines an fp32 tensor with a Python float, torch's wrapped-scalar machinery computes the scalar in float64 and casts it ONCE to fp32 — it does NOT pre-round the factors to fp32 and then combine. A host that derives coefficient scalars through fp32 arithmetic reproduces the wrong chain: `float32(1.0-0.9) != float32(1.0) - float32(0.9)` (1 ULP). Derive such scalars in C++/Python DOUBLE, then cast each ONCE to the kernel's fp32 scalar type.

**Concrete anchor**:
```cpp
// derived AdamW bias-correction divisors: compute in double, single-cast
const double d1 = 1.0 - std::pow((double)beta1, (double)step);
const float  dM1_ = (float)d1;   // ONE rounding — mirrors torch float64→fp32
```

**Evidence**: 17_AdamW 310p (2026-08-15, probed): switching the derived (1−β1^t) / (1−β2^t) chains from fp32-step arithmetic to double-then-single-cast was load-bearing for bit-exactness of the m/v outputs (MERE 0.0; `17_AdamW/knowledge_update.md` finding 5).

**Other instances (predicted)**: any op with host-derived coefficient chains — optimizer hyperparameters (Adam/AdamW bias corrections, momentum decays), LR schedules, normalization (1−momentum), weight-decay factors — wherever the reference is a Python-level formula applied to fp32 tensors.

**Cross-ref**: P-P51 (the IN-KERNEL scalar-chain native-dtype rule — a different level: P-P51 governs arithmetic dtype inside the kernel against a native-dtype reference chain; this entry governs HOST-side derivation of scalar constants against torch's float64 wrapped-scalar semantics; both can bind the same op at different sites). backend=all.
