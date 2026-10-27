---
type: Runbook
title: 'Synthesize the cancellation-free identity by range reduction when the platform ships no primitive'
status: draft
description: 'exp(x)−1 / log1p / 1−cos / √(1+x)−1 / ELU-CELU-SELU-softplus 负分支对 CPU 真值评级时，AscendC 无 Expm1/Log1p 原语：用 range reduction 合成（k=rint(t·log2e)，二部 ln2，degree-8 Horner），branch-free'
sources: []
tags:
- task.design
- topic.precision
platforms:
- '950'
aliases: ['P-P200']
created_at: '2026-09-16T00:00:00Z'
updated_at: "2026-10-15T00:00:00Z"
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P200 — P-P200: A5Ops additive extension

**P-P200** — **when the cancellation-free identity has NO library primitive, SYNTHESIZE it by range reduction.** Fires on `exp(x)−1` / `log1p` / `1−cos` / `√(1+x)−1` / ELU-CELU-SELU-softplus negative branches graded vs CPU truth, where AscendC ships no `Expm1`/`Log1p`. Recipe: `k=rint(t·log2e)`, `r=t−k·ln2_hi−k·ln2_lo` (2-part ln2 keeps `k·ln2_hi` exact), `P=expm1(r)` degree-8 Horner, `expm1(t)=2^k·P+(2^k−1)`. **Branch-free** because the dangerous band `\

- Former A5Ops ID: `P-P200`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR/Ascend950DT (arch 351x); cann=9.0.0; bisheng=n/a; op_class=elementwise activation / any kernel containing exp(x)−1, log1p, 1−cos, √(1+x)−1; dtype=fp32-internal`
`verified_on: soc=Ascend950DT_9582; cann=9.0.0 (celu kw-1, 2026-08-02 — 156/156 T1 PASS, det 156/156 bit-identical)`
`unverified_on: soc=Ascend910_220x (A3 — the identity is arithmetic and should transfer, but the Cast/RINT and subnormal behaviours it leans on are per-arch; re-measure)`

**Trigger**: the kernel computes a near-cancellation expression (`exp(x)−1`, `log(1+x)`,
`1−cos x`, `√(1+x)−1`, an ELU/CELU/SELU/softplus negative branch), it is graded against CPU
truth, **and** the platform ships no complementary primitive. On AscendC: **there is no
`Expm1` and no `Log1p`** — confirm by grepping `API_CATALOG.md` (OL-80) before assuming.
OL-271 is the entry-point decision rule; this is its "must synthesize" branch.

**Do not settle for `Exp(x) − 1`, and do not `Select` between a small-|x| series and a
large-|x| naive form.** A branch-free range-reduced form exists and is strictly better than
both.

**The recipe** (stated for `expm1`; the same shape works for `log1p` via its own reduction):

```
k = rint(t · log2e)                      // integer, as a float
r = t − k·ln2_hi − k·ln2_lo              // r ∈ [−0.347, 0.347]; 2-part ln2 keeps k·ln2_hi EXACT
P = expm1(r)  via Horner, degree ~8      // relatively accurate — a polynomial IN r, no subtraction
expm1(t) = 2^k · P + (2^k − 1)
```

**Why it is branch-free — the load-bearing argument.** The dangerous band is exactly
`|t| ≤ 0.347`, and there `k = 0`, so `2^k = 1` and the expression **collapses algebraically to
`P(t)`** — a polynomial in `t`, relatively accurate, containing no near-equal subtraction. For
`k ≤ −1` both terms are O(1) and `|sum| ≥ 0.29`, bounding amplification at ~3.4×. So one
straight-line expression is correct across the whole domain; the regime split that would
normally need a `Select` is absorbed by the reduction itself. (`Select` on a VEC pipe costs a
mask + both branches evaluated; avoiding it is a bonus, not the point.)

**Companion technique — build `2^k` with float MULTIPLY, never `ShiftLeft`.** Place `(k+127)`
into the IEEE exponent field by multiplying by `2²³` and casting, keeping the whole chain in
float ops. This avoids int32 binary-scalar arithmetic (**PB-23**) and drops a `ShiftLeft`
signature dependency. Exact because `(k+127)·2²³ ≤ 253·2²³ = 2.12e9 < INT32_MAX` and the value
carries only 8 significant mantissa bits. Reusable in ANY software `exp`/`expm1`/`pow`:

```cpp
AscendC::Adds(kf, kf, 127.0f, n);              // biased exponent, still float
AscendC::Muls(kf, kf, CELU_EXP_SHIFT, n);      // ×2^23 == "<< 23", in float
LocalTensor<int32_t> ei = r.template ReinterpretCast<int32_t>();
AscendC::Cast(ei, kf, RoundMode::CAST_RINT, n);  // exact; r now holds the bits of 2^k
```

Use `CAST_RINT` (IEEE round-to-nearest-even) for the `rint(k)` step — `CAST_ROUND` is
round-half-UP and biases `k` on exact `.5` boundaries (**OL-81**).

**Promote fp32-internally for BOTH dtypes.** Running the chain at fp32 even on the fp16 path is
what makes fp16 clean: a native-fp16 `exp(t)−1` carries ~0.5 fp16 ULP (4.9e-4) of absolute
error into a result that can itself be ~1e-3. It also **structurally immunises the op against
the fp16 scalar-attr narrowing trap** (`static_cast<half>(alpha)`, where `fp16(0.7) =
0.699951`) — there is no narrowing to get wrong, because the attrs stay fp32 scalars. Cast only
at the final store, with `CAST_RINT`.

**Measured** (celu kw-1, 2026-08-02, A5 Ascend950DT_9582, port_a3_to_a5, CPU-truth): `156/156`
T1 PASS, determinism `156/156` bit-identical, build PASS on the first compile, **0
precision-fix iterations** — the precision work was front-loaded into analysis, before any
kernel line was written. Pre-hardware numerical validation over 200k samples in `t ∈ [−0.01,
0)`: max rel err of `exp(t)−1` = **1.0** with a correctly-rounded fp32 `exp`, vs **8.8e-8**
(~¾ fp32 ULP) with this form. **Positive control (the load-bearing half):** the same kernel
with hardware `Exp` substituted for the software `expm1` scored **59/156** — so the green is
attributable to this pattern, not to the dataset.

**Anti-pattern boundary**: do NOT reach for this when the primitive itself is the problem —
that is P-P88's Cephes rewrite (raising primitive *grade*) or OL-103's tiering. This pattern
addresses **operand ordering**, an orthogonal axis; a better `exp` provably cannot fix
`exp(x)−1` (OL-271 Step 0). Conversely, if a complementary primitive DOES exist (`Erfc` for
`1+erf`), just call it — do not synthesize.

**Determinism**: pure Mul/Add/Adds/Muls/Cast/Duplicate — the OL-103 Tier-1 bit-exact palette
only, no hardware transcendental. Deterministic by construction (A-P61).

**Cross-ref**: OL-271 (the decision rule that routes here), OL-403 (why the near-zero grading
band this fixes is not a free pass), OL-103 (Tier-1 palette + software-fp32 recipe), OL-80
(grep API_CATALOG before concluding a primitive is absent), OL-81 (CAST_RINT vs CAST_ROUND),
PB-23 (int32 binary-scalar arithmetic — what the float-multiply exponent build avoids), P-P88
(the OTHER axis: primitive grade). backend=ascendc.

---

## 适用场景

- **工程形态**：exp(x)−1 / log1p / 1−cos / √(1+x)−1 / ELU-CELU-SELU-softplus 负分支对 CPU 真值评级时。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
