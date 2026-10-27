---
type: Runbook
title: "`Muls(dst, src, tensorOperand, n)` fails to compile — the `-s`-suffix VEC ops take a SCALAR operand; the tensor×tensor product is `Mul`"
description: "`Muls` 是标量广播变体，第三个参数传 LocalTensor 是类型错误——张量×张量要用 `Mul`。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-291", "Muls", "Mul"]
---

# `Muls(dst, src, tensorOperand, n)` fails to compile — the `-s`-suffix VEC ops take a SCALAR operand; the tensor×tensor product is `Mul`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=n/a (API-surface contract); op_class=all (any VEC arithmetic chain)`
- **Symptom**: compile error on a line like `Muls(p, p, u, un)` where `u` is a `LocalTensor` — `Muls` is the scalar-broadcast variant (`Muls(dst, src, scalarValue, count)`), so a tensor 3rd argument is a type error. The tensor×tensor elementwise product is `Mul(dst, a, b, count)`.
- **Rule**: the `-s`-suffix family (`Muls`/`Adds`/`Subs`/`Divs`/...) broadcasts a SCALAR; the bare forms (`Mul`/`Add`/`Sub`/`Div`) are tensor×tensor. Same API-shape distinction EC-80 documents from the other side (scalar semantics misread via `Duplicate` block-broadcast); PB-23 lists the dtype rejects of the scalar family on V220. <!-- terminology-ok: verbatim citation of pre-existing KB entry id -->
- **Concrete anchor**: `代码片段 33_CTClossV3/ctclossv3_kernel.h.snippet.h（见文末）` — the software-exp Horner step: `Mul(p, p, u, un)` for the tensor operand (the reduced argument), `Muls`/`Adds` only where the operand is a literal polynomial coefficient.
- **Evidence**: 33_CTClossV3 310p (2026-08-22, failures_ledger build iter 1 `33_CTClossV3/failures_ledger.md:5`): typo'd `Muls(p, p, u, un)` in the expm1 Horner; switching that one call to `Mul` fixed it. Cost: 1 compile iteration.
- **Other instances (predicted)**: any hand-written polynomial/elementwise chain where a coefficient VECTOR is staged in UB (then it is `Mul`) vs a literal constant (`Muls`) — easy to slip when transcribing a scalar math formula into VEC form. backend=ascendc.

## 内嵌代码片段

### 33_CTClossV3/ctclossv3_kernel.h.snippet.h

```cpp
 * e=1.0 bit-exactly (d=0 path), so walk is ~sqrt(T)*delta ~ 5e-5 absolute on
 * loss magnitude >= ~1 -> rel ~5e-5 << 1.22e-4 (MERE) with ~2.4x margin.
 */

namespace ctclossv3_ops {

// Capacity (dataset maxima: L <= 251 -> S' = 2L+1 <= 503; C <= 500 -> C8 <= 512)
static constexpr int32_t S2_MAX = 512;
static constexpr int32_t C8_MAX = 512;

// Multi-sample lane batching (ko-3 2026-08-22): each block processes a GROUP
// of samples per t-step instead of one sample at a time. Per-sample s-vectors
// occupy 64-lane-aligned SEGMENTS of the shared S2_MAX-lane vectors; the DP
// math is lane-wise identical (bit-identical outputs), but the per-step
// dependency-stall cost (Opt3 falsification: latency-bound serial recurrence,
// lanes mostly idle at S2 << 512) is amortized over the group.
static constexpr int32_t LANE_UNIT = 64;              // VSEL/Compare granularity
static constexpr int32_t LANE_UNITS = S2_MAX / LANE_UNIT;  // 8 units of 64
static constexpr int32_t GRP_MAX = LANE_UNITS;        // >=1 segment per sample
__aicore__ inline float BitsToF32(uint32_t u) { F32Bits c; c.u = u; return c.f; }
__aicore__ inline uint32_t F32ToBits(float f) { F32Bits c; c.f = f; return c.u; }

// Scalar fp32 exp for the final 2-element combine — same math as the VEC path.
__aicore__ inline float ScalarExp(float x) {
    float kf = x * EXP_LOG2E;
    // floor(kf + 0.5) == rint for our domain kf in [-125.7, 0]
        // lengths): target min(N, NB) groups so N <= NB degenerates to the
        // original one-sample-per-block shape; N > NB packs ~N/NB samples per
        // group, closing early when the 512-lane budget would overflow.
        const int32_t nTarget = static_cast<int32_t>(
            nb < N_ ? (nSort < nb ? nSort : static_cast<int32_t>(nb)) : nSort);
        const int32_t per = nTarget > 0 ? (nSort + nTarget - 1) / nTarget : 1;
        int32_t gFirst[N_SORT_MAX + 1];
        int32_t nGroups = 0;
        {
            int32_t i = 0;
            while (i < nSort) {
                int32_t units = (2 * llenArr[order[i]] + 1 + 63) >> 6;
                int32_t cnt = 1;
                gFirst[nGroups++] = i;
        auto idx = bufIdx_.Get<uint32_t>();
        auto logp = bufLogp_.Get<float>();
        Gather<float>(logp, row, idx, 0, un);
        rowQue_.FreeTensor(row);
    }

    // element-wise fp32 copy helper (count elements)
        Mul(p, p, u, un); Adds(p, p, EXP_C5, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C4, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C3, un);
        Mul(p, p, u, un); Adds(p, p, EXP_C2, un);
        Mul(p, p, u, un); Adds(p, p, 1.0f, un);      // bracket 1 + u/2 + ... + u^8/8!
        Mul(p, p, u, un);                            // p = expm1(u)
        Adds(p, p, 1.0f, un);                        // p = e^u
        // 2^k via the celu-proven float bit trick
        Adds(k, k, EXP_K_BIAS, un);
```
