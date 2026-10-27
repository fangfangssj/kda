---
type: Runbook
title: "Host-derived scalar coefficients: compute in double, cast once to the tensor dtype"
description: "Host-derived scalar coefficients: compute in double, cast once to the tensor dtype"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-550 — OL-550: A5Ops additive extension

> 关联：同一 OL-550 主题已由既有卡 [ol_550_host_derived_scalar_coefficients_compute_in_doub.md](ol_550_host_derived_scalar_coefficients_compute_in_doub.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all (host/pybind-side numerics)。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

**OL-550** — **host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype** — torch wrapped-scalar semantics feed float64→fp32-cast scalars, so fp32-step-derived chains are 1 ULP off (`float32(1.0-0.9) != float32(1.0)-float32(0.9)`). Binds optimizer bias corrections, momentum decays, LR schedules, 1−momentum factors — any Python-level formula applied to fp32 tensors. 17_AdamW 310p 2026-08-15: load-bearing for m/v bit-exactness (MERE 0.0). Different level from P-P51 (in-kernel chain dtype — both can bind the same op at different sites). backend=all.

- Former A5Ops ID: `OL-550`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (host/pybind-side numerics)`
<!-- applies_to_backend: all -->

**Principle**: when a PyTorch reference combines an fp32 tensor with a Python float, torch's wrapped-scalar machinery computes the scalar in float64 and casts it ONCE to fp32 — it does NOT pre-round the factors to fp32 and then combine. A host that derives coefficient scalars through fp32 arithmetic reproduces the wrong chain: `float32(1.0-0.9) != float32(1.0) - float32(0.9)` (1 ULP). Derive such scalars in C++/Python DOUBLE, then cast each ONCE to the kernel's fp32 scalar type.

**Concrete anchor**:
```cpp
// derived AdamW bias-correction divisors: compute in double, single-cast
const double d1 = 1.0 - std::pow((double)beta1, (double)step);
const float  dM1_ = (float)d1;   // ONE rounding — mirrors torch float64→fp32
```

**Evidence**: 17_AdamW 310p (2026-08-15, probed): switching the derived (1−β1^t) / (1−β2^t) chains from fp32-step arithmetic to double-then-single-cast was load-bearing for bit-exactness of the m/v outputs (MERE 0.0; `workspace/17_AdamW/knowledge_update.md` finding 5).

**Other instances (predicted)**: any op with host-derived coefficient chains — optimizer hyperparameters (Adam/AdamW bias corrections, momentum decays), LR schedules, normalization (1−momentum), weight-decay factors — wherever the reference is a Python-level formula applied to fp32 tensors.

**Cross-ref**: P-P51 (the IN-KERNEL scalar-chain native-dtype rule — a different level: P-P51 governs arithmetic dtype inside the kernel against a native-dtype reference chain; this entry governs HOST-side derivation of scalar constants against torch's float64 wrapped-scalar semantics; both can bind the same op at different sites). backend=all.

---