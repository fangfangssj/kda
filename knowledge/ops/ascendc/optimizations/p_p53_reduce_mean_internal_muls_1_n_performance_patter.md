---
type: Optimization
title: "reduce_mean internal Muls(1/N) — **performance pattern, not a precision requirement**"
description: "Major correction (2026-04-16 evening): minimal repro confirmed that tensor / N and tensor (1/N) are bit-identical on NPU (both fp32 and bf16, across all N values). The original P-P53 inference that \"p"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p53-reduce-mean-internal-muls-1-n-performance-patter.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# reduce_mean internal Muls(1/N) — **performance pattern, not a precision requirement**

## 优化点

**Major correction (2026-04-16 evening)**: minimal repro confirmed that `tensor / N` and `tensor * (1/N)` are **bit-identical** on NPU (both fp32 and bf16, across all N values). **The original P-P53 inference that "precision differs" is wrong**.

**Trigger**: kernel internally implements a `.mean()` aggregation (sum then divide by count to get the mean).

**Pattern**: CANN's `reduce_mean_dag.h`:
```cpp
CopyIn → Cast<fp32> → ReduceSumOp<fp32> → Muls<fp32>(1/N) → Cast<T> → CopyOut
//                                        ^^^^^^^^^^^^^^^
//                                   Muls with pre-computed reciprocal
```

**Correct interpretation**: CANN uses Muls(1/N) for **performance** (pre-divide once vs divide per element), not for precision — the two are bit-equal on NPU.

**Do NOT conclude that "Muls(1/N) can always replace Div(N)"**:
- In the CANN reduce_mean DAG scenario — "divide by N immediately after sum is produced" — the two are equivalent (bit-identical).
- But in complex chained-division scenarios like op #14, **forcing the substitution can introduce other bugs** (see lesson below).

**Counter-example (misuse)** (the `grad_mean / spatial_size` term in the grad_input formula of op #14):
```python
# reference (PyTorch):

> 原 OKF v1 卡号：**P-P53**。
grad_input = ... + grad_mean / spatial_size  # ordinary tensor / int → tensor Div
```
- kernel should use `Divs(grad_mean_tensor, spatial_float)` or `Duplicate(spatial)+Div` → tensor Div path
- should **not** be rewritten as `Muls(grad_mean_tensor, invSpatial)` — that is the internal form of reduce_mean
- Lesson (2026-04-16): generalizing `/spatial` to `*invSpatial` broke fp32-case precision.

**Example (correct use)** (if the kernel implements `.mean()` itself):
```cpp
// simulate tensor.mean() behavior:
ReduceSumP47(src, count);  // sum over all elements
// then:
Muls(src, src, invN, count);  // matches CANN reduce_mean DAG
// not:
Divs(src, src, float(count));  // not the internal form of CANN mean
```

**Detection**: ask yourself — is this division **part of an aggregation** (inside `.mean()`) or an **ordinary tensor/scalar operation**?
- inside aggregation → Muls(1/N)
- general division → Div / Divs (preserve tensor Div semantics)

**Related trap**: P-P55 (Pow) has a similar issue — do not generalize the internal implementation of Pow (`Exp(Ln(x)*y)`) to all pow scenarios; if you need `pow(x, 3)`, `x*x*x` may actually be closer to CANN's Pow implementation.

**Source code path** (`reduce_mean_dag.h:28-39`): `Vec::Muls<PromteT>(reciprocal_constant)`

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P53，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `performance`；trigger: Target CANN uses `Muls(1/N)` to precompute the reciprocal once, and an A5 minimal repro found `tensor / N` and `tensor * (1/N)` bit-identical for fp32 and bf16 across the tested N values. This does not override selected-source authority: the final choice must match OL-97 selected-source behavior or CPU fp64 truth; target CANN is advisory, and outside the immediate post-reduction mean DAG the reference's `Div`/`Divs` semantics must be preserved.；优先级: LOW。

> 索引行原文：`| P-P53 | reduce_mean internal Muls(1/N) — performance pattern, not a precision requirement | performance | Target CANN uses `Muls(1/N)` to precompute the reciprocal once, and an A5 minimal repro found `tensor / N` and `tensor * (1/N)` bit-identical for fp32 and bf16 across the tested N values. This does not override selected-source authority: the final choice must match OL-97 selected-source behavior or CPU fp64 truth; target CANN is advisory, and outside the immediate post-reduction mean DAG the reference's `Div`/`Divs` semantics must be preserved. | LOW |`
