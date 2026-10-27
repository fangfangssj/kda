---
type: Runbook
title: "When the cancellation-free identity has NO library primitive, SYNTHESIZE it by range reduction — the reduced form is branch-free precisely because `k=0` covers the dangerous band"
description: "软件 exp 里用 Adds(k,127)+Muls(2^23) 再 reinterpret 构造 2^k 替代 ShiftLeft；两条 dtype 路径都内部升 fp32 计算、最后才 CAST_RINT 落盘——精度工作前置到分析阶段"
tags: [task.optimization, topic.compilation, topic.compute.reduction]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P272"]
---

# When the cancellation-free identity has NO library primitive, SYNTHESIZE it by range reduction — the reduced form is branch-free precisely because `k=0` covers the dangerous band

## 优化点

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR/Ascend950DT (arch 351x); cann=9.0.0; bisheng=n/a; op_class=elementwise activation / any kernel containing exp(x)−1, log1p, 1−cos, √(1+x)−1; dtype=fp32-internal`
`verified_on: soc=Ascend950DT_9582; cann=9.0.0 (celu kw-1, 2026-08-02 — 156/156 T1 PASS, det 156/156 bit-identical)`
`unverified_on: soc=Ascend910_V220 (A3 — the identity is arithmetic and should transfer, but the Cast/RINT and subnormal behaviours it leans on are per-arch; re-measure)`
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

**Evidence (2nd platform — dav-m200, quant-scale-amplified cancellation)**: swi_glu_quant 310p
(2026-09-02, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN, CPU-truth
golden, 35/35 PASS, det 5/5) — an int8-quant SwiGLU whose per-row dynamic scale (`127/|amax|`,
up to ~1e12) AMPLIFIES any residual: the identity shortcut `sigmoid(z) = 1 − sigmoid(|z|)` for
`z<0` lost `e^{-|z|}` below ~1e-7 and, after the scale multiply, corrupted the quantized rows
with large-`|z|` entries. Fix: compute `sig = (z≥0) ? 1/(1+w) : w/(1+w)` with `w = e^{-|z|}`
— never evaluate the complement branch as a subtraction. On the VEC path `w` is obtained as
`1/e^{+|z|}` (`Reciprocal` of the exp), reusing this entry's float-multiply `2^k`
reconstruction. The scalar path surfaced the companion pitfall in that reconstruction: build
the EXPONENT BITS (`(k+127)<<23`, or `Muls((k+127), 1<<23)` then `Cast<int32, RINT>`), never
compute `(k+127)·2^23` as a float VALUE — the value-space product reinterpreted as bits yields
a wrong-power `2^k` → garbage through the sigmoid chain. Generalizes this entry's
identity-choice principle to any cancellation whose residual feeds a large downstream
multiply (quant scale, denormalizer); P-P88 covers the same identity from the
primitive-grade axis. Anchor: `swi_glu_quant/knowledge_update.md` finding 5 +
`swi_glu_quant/failures_ledger.md` precision iters 4–5.
**Anti-pattern boundary**: do NOT reach for this when the primitive itself is the problem —
that is P-P88's Cephes rewrite (raising primitive *grade*) or OL-103's tiering. This pattern
addresses **operand ordering**, an orthogonal axis; a better `exp` provably cannot fix
`exp(x)−1` (OL-271 Step 0). Conversely, if a complementary primitive DOES exist (`Erfc` for
`1+erf`), just call it — do not synthesize.
**Determinism**: pure Mul/Add/Adds/Muls/Cast/Duplicate — the OL-103 Tier-1 bit-exact palette
only, no hardware transcendental. Deterministic by construction (A-P61).
**Cross-ref**: [`OL-271`](../precision/ol_271_gaussian_cdf_tail_erfc_direct_form.md) (the decision rule that routes here), [`OL-560`](../compilation/ol_560_one_consuming_gm_write_per_tque_enque_deque_cycl.md) (why the near-zero grading
band this fixes is not a free pass), OL-103 (Tier-1 palette + software-fp32 recipe), OL-80
(grep API_CATALOG before concluding a primitive is absent), OL-81 (CAST_RINT vs CAST_ROUND),
PB-23 (int32 binary-scalar arithmetic — what the float-multiply exponent build avoids), P-P88
(the OTHER axis: primitive grade). backend=ascendc.
