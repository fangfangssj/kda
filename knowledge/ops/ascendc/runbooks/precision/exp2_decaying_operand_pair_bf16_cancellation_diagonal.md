---
type: Runbook
title: "A decayed/undecayed operand pair (exp2(+g) x exp2(-g)) destroys its own exact cancellation once both factors are stored in bf16"
description: "A decayed/undecayed operand pair (exp2(+g) x exp2(-g)) destroys its own exact cancellation once both factors are stored in bf16"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-295"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-295 — OL-295: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR (measured on _958b); arch=3510; cann=9.1.T500; op_class=linear-attention / gated-delta-rule / any kernel forming a product from exp2(+g) and exp2(-g) operands stored in a low-precision dtype。
- 验证来源：device, FlashKDA fwd, case shape_d128_t16_initial_stateF -- stage-by-stage diff of the kernel's own workspace against the fp64 reference。

**OL-295 — a decayed/undecayed operand PAIR (`exp2(+g)` x `exp2(-g)`) destroys its own exact cancellation once both factors are stored in bf16**. The diagonal of `q*exp2(+g) @ (k*exp2(-g))^T` is decay-free in exact math (`exp2(g_i-g_i)=1`) but the kernel multiplies two SEPARATELY rounded factors spanning 2^+-115, so the diagonal is wrong and degrades with row index (measured 35x by row 3; stage MERE 0.0959 vs operands ~0). **The strictly-lower sibling matmul hides it** — `tril(...,-1)` discards the diagonal, so a clean `L` is NOT evidence the pairing is safe. Detection: diff the workspace buffer against fp64 and split diagonal vs off-diagonal. Fixes: compute the decay-free diagonal in fp32 and overwrite; or re-centre the exponent (`g_i-g_ref`, `g_ref-g_j`) so both factors stay near 1. [arch35, Ascend950PR, linear-attention, gated-delta-rule, bf16, exp2, decay, cancellation, cube-operand, diagonal]

- Former A5Ops ID: `OL-295`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR (measured on _958b); arch=3510; cann=9.1.T500; op_class=linear-attention / gated-delta-rule / any kernel forming a product from exp2(+g) and exp2(-g) operands stored in a low-precision dtype`
`verified_on: device, FlashKDA fwd, case shape_d128_t16_initial_stateF -- stage-by-stage diff of the kernel's own workspace against the fp64 reference`
**The shape.** Chunked linear attention builds two operands from the same gate:
`k_decayed = k * exp2(g_cumsum)` and `k_inv = k * exp2(-g_cumsum)`, then forms
`M = q_decayed @ k_inv^T`. Element (i,j) is
`sum_d q[i,d]*k[j,d]*exp2(g[i,d] - g[j,d])`, so **the DIAGONAL is decay-free**:
`exp2(g_i - g_i) = 1` exactly, and M[i,i] should be `q_i . k_i` regardless of the gate.
**What goes wrong.** The kernel does not compute `exp2(g_i - g_j)`; it multiplies two
SEPARATELY ROUNDED factors. With `lower_bound = -5` the cumulative gate reaches
**-115**, so the factors span 2^+-115 -- each is representable in bf16 (same exponent
range as fp32) but each carries its own ~2^-9 relative rounding, and their product no
longer cancels. The diagonal, which should be exact, degrades **with row index**
because |g| grows along the chunk.
**Measured (vendor MERE, per-stage, kernel vs fp64):**
| stage | kernel | bf16 floor | note |
|---|---|---|---|
| `k_decayed`, `q_decayed`, `k_inv` | ~0 | -- | operands themselves correct |
| **`mraw` = q_decayed @ k_inv^T** | **0.0959** | -- | **defect** |
| `lraw` = k_decayed @ k_inv^T | 0.0038 | -- | ok, see why below |
    mraw mean |err| on the diagonal      1.8e-05      <- looks fine
    mraw mean |err| off the diagonal     8.2e+19      <- on values ~1e34, so relatively tiny
    kernel diag[0..3]  -0.000594  0.013999  -0.013990  -3.5e-05
    fp64   diag[0..3]  -0.000586  0.014017  -0.013976   +1.0e-06
                                                        ^^^^^^^ 35x wrong by row 3
**Why the sibling matmul hides it.** `L = tril(k_decayed @ k_inv^T, -1)` is STRICTLY
lower, so it discards the diagonal, and its kept entries have `g_i - g_j < 0` and are
exponentially small -- absolute error there is negligible. `Mqk = tril(q_decayed @
k_inv^T)` KEEPS the diagonal, where the value is O(1). **Two matmuls with the same
operands and the same defect, and only the one that keeps the diagonal shows it.** Do
not conclude from a clean `L` that the pairing is safe.
**Detection.** Diff the kernel's own workspace buffer against fp64 and split the error
by diagonal vs off-diagonal. A large off-diagonal absolute error on huge values is
NORMAL here (the upper triangle legitimately reaches 2^115 and is masked away later);
the signal is a diagonal that should be exact and is not, degrading with index.
**Fix directions** (not yet device-verified -- recorded as options, not as a result):
1. **Compute the decay-free diagonal separately** in fp32 and overwrite it after the
   matmul. Cheapest, and exactly targets the provably-exact entries.
2. **Re-centre the exponent** so both factors stay near 1: use `exp2(g_i - g_ref)` and
   `exp2(g_ref - g_j)` with a per-tile or per-row `g_ref`. The product is unchanged
   mathematically while each stored factor loses far less bf16 range. This is the same
   manoeuvre FlashAttention's max-subtraction performs for a different reason.
3. Keep one side of the pair in fp32 where the consumer allows it -- **not available
   when both sides are cube matmul operands**, which is the case here (see
   CAND-SCAN-FP32-ACCUM's scope boundary for the same constraint).
**Cross-ref**: CAND-SCAN-FP32-ACCUM (its scope boundary: fp32 carry is useless when
the value is a cube operand), OL-294 (arch35 AIV work-split), EC-26 (unaligned VEC
offsets in the same op's mask kernel).
---