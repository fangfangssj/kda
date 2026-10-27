---
type: Optimization
title: "Divs(vec,scalar) → Muls(vec,1.0f/scalar) is precision-safe in fp32 and 4-8× faster on da Vinci VEC"
description: "In fp32 a*(1/b) differs from a/b by ≤1 ULP, so replacing Divs with a precomputed-reciprocal Muls is precision-safe and 4-8× faster; keep Divs for fp16/bf16 where 1/b loses mantissa."
tags: [task.optimization, topic.precision, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/ol-256-divs-to-muls-fp32-precision-safe.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
---

# Divs(vec,scalar) → Muls(vec,1.0f/scalar) is precision-safe in fp32 and 4-8× faster on da Vinci VEC

> 原 OKF v1 卡号：**OL-256**。

## 优化点 / 选型
**When to use:** any op with a scalar-divide normalizer (RMSNorm / LayerNorm / GroupNorm / instance-norm / fused-norm-quant) whose kernel-internal compute is fp32.

**Principle.** da Vinci VEC `Divs(vec, scalar)` throughput ≈ 1 elem/cycle; `Muls(vec, scalar)` ≈ 8-16 elem/cycle. Precomputing `inv = 1.0f / scalar` (ONE scalar divide, not pipelined through VEC) and replacing `Divs(vec, scalar)` with `Muls(vec, inv)` is a **4-8× speedup on the normalization step with zero precision cost in fp32**.

**Why it's precision-safe in fp32.** In IEEE-754 fp32 both `a/b` and `a*(1/b)` are correctly rounded — the intermediate `1/b` rounds once, each `a_i * (1/b)` rounds once, and each `a_i / b` also rounds once; the two sequences differ by at most 1 ULP. This is **fundamentally different from fp16**, where `1/b` loses 13+ bits of mantissa and `a*(1/b) ≠ a/b` is a real precision hazard.

**Decision rule:**
- **fp32 internal compute: MUST use `Muls(x, 1.0f/scalar)`** — provided every
  consumer of the result is **continuous**.
- **fp16/bf16 internal compute: keep `Divs`.**
- When in doubt: the first iteration may use `Divs` for safety; the optimizer converts to `Muls` behind a precision gate.

**Scope precondition (added 2026-09-25, crop_and_resize counter-evidence).** The ≤1 ULP
bound holds for the isolated transform. A **discontinuous or
cancellation-prone consumer** can amplify that 1 ULP to full scale: result
fed into `floor()`/`ceil()`/`round()`/comparison, or into near-equal-magnitude
subtraction such as `y_ratio = in_y - floor(in_y)` — a 1-ULP input flip across
an integer boundary changes the fractional part by ~1.0. Measured: a
crop_and_resize fp32 kernel that applied the MUST rule verbatim produced
exactly one failing case (case 48, matched_ratio 0.9831 < 0.99); reverting that leg to `Divs` restored 50/50. Before applying the
MUST rule, trace the result's consumers: any floor/ceil/round/compare or
near-equal subtraction → keep `Divs` on that leg.

### Concrete anchor
```cpp
// BEFORE (V1, geo_mean 0.47×):
float rms = sqrt(meanSq + eps);
Divs(work, work, rms, N);
PipeBarrier<PIPE_V>();

// AFTER (V2, geo_mean 0.91×):
float rms = sqrt(meanSq + eps);
float invRms = 1.0f / rms;
Muls(work, work, invRms, N);
PipeBarrier<PIPE_V>();
```

### Evidence
add_rms_norm_quant V1→V2 A/B (Ascend950PR_957b, CANN 9.0.0, NPU 0, 196 cases, 2026-06-24):
- V1 (Divs): geo_mean 0.47×, arith_mean 0.50×
- V2 (Muls): geo_mean 0.91×, arith_mean 1.04×
- Precision: V2 196/196 PASS (better than V1's 194/196); x_out max_diff ≤ fp32 1 ULP vs CPU fp64 truth.
- The Divs→Muls transform contributed ~45% of the 1.94× speedup.

### Related
- ALWAYS_LOADED_RULES §5 fp32 carve-out (normative permission), OL-257 (VEC cost model — the "why it's faster" half), OL-245 (regbase — same "prefer the efficient API" principle).
