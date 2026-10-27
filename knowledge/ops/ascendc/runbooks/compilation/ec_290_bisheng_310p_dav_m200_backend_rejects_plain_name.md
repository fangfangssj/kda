---
type: Runbook
title: "bisheng (310p/dav-m200 backend) rejects plain namespace-scope `constexpr` in aicore code — use `static constexpr` (internal linkage)"
description: "310p bisheng 下 namespace 作用域的裸 `constexpr int32_t` 不编译，加 `static`（内部链接）即可。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-290"]
---

# bisheng (310p/dav-m200 backend) rejects plain namespace-scope `constexpr` in aicore code — use `static constexpr` (internal linkage)

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; bisheng=(310p lane, version unrecorded); op_class=all (aicore TU compile)`
- **Symptom**: a plain `constexpr int32_t X = ...` at namespace scope in kernel device code fails to compile under the 310p bisheng backend; the same declaration with `static` (internal linkage) compiles clean. The downstream fatal can be a confusing cascade (e.g. "global constant not allowed") rather than a pointer at the missing `static`.
- **Fix / rule**: every namespace-scope constant in an aicore TU on this lane is `static constexpr` — the convention already used across the 310p lane's earlier kernels (23_RepeatInterleave / 24_EmbeddingDenseBackward / 25_NLLLoss era).
- **Concrete anchor**: `代码片段 33_CTClossV3/ctclossv3_kernel.h.snippet.h（见文末）` — all constants (`S2_MAX`, `C8_MAX`, `U32_NEG_INF`, `U32_POS_INF`, `U32_NAN`, `EXP_LOG2E`, ...) declared `static constexpr`.
- **Evidence**: 33_CTClossV3 310p (2026-08-22, benchmark, failures_ledger build iter 3 `33_CTClossV3/failures_ledger.md:11`): iter-3 build rejected the plain form; the `static` form is in the shipped kernel (Pass A 100/100 + Pass B 236/236 + det 100/100, DEVICE-RUN).
- **Evidence (2nd op — macro alternative)**: swi_glu_quant 310p (2026-09-02, benchmark-acceptance lane, failures_ledger build iter 1 `swi_glu_quant/failures_ledger.md`): namespace-scope `constexpr float` in the kernel header hit the same "global constant not allowed in aicore"; this worker's fix used preprocessor `#define` macros (shipped 35/35, DEVICE-RUN). `static constexpr` (this entry) is the preferred internal-linkage form; macros are a working alternative where the constant must survive in a macro-expansion context.
- **Other instances (predicted)**: any new dav-m200 kernel header that copies a constexpr constant block from a host-side or A5-side template verbatim. backend=ascendc.

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
