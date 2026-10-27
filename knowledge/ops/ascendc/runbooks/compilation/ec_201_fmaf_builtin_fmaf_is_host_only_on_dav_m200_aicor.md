---
type: Runbook
title: "fmaf / __builtin_fmaf is host-only on dav-m200 aicore (compile \"host function from aicore\" + link undefined sy"
description: "dav-m200 aicore 上 `fmaf`/`__builtin_fmaf` 是 host-only——没有单舍入 FMA，分离 mul+add 对硬件 FMA 参考会差 ~1 ULP，要诚实声明精度目标。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-201"]
---

# fmaf / __builtin_fmaf is host-only on dav-m200 aicore (compile "host function from aicore" + link undefined sy

**EC-201** — `fmaf` / `__builtin_fmaf` is **host-only on dav-m200 aicore** (compile "host function from aicore" + link `undefined symbol: fmaf`); no single-rounding FMA for accumulation, and NO SIMT scalar escape hatch on arch200 ([`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md) is arch35-only). Separate mul+add → ~1-ULP fp16 drift vs a hardware-FMA reference (MEASURED, not waived — bit-exact vs CANN is precluded by the rounding path; declare the target honestly). Evidence: incre_flash_attention 27/27 ≤1-ULP vs 910B2C golden while CANN `aclnnIncreFlashAttention` is bit-exact cross-platform. Cross-ref CAND-A5OPS-0004 (transcendentals, sibling), [`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md), CAND-DAV-M200-UB-FREE-SCALAR-KERNEL. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all (any aicore accumulation that wants IEEE-754 single-rounding fused-multiply-add)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — incre_flash_attention`
`unverified_on: soc=Ascend950PR, Ascend910 (FMA availability is arch/compiler-dependent — A5 may expose a SIMT FMA intrinsic via the OL-242-style scalar path; re-verify before assuming FMA is unavailable everywhere)`

- **Error pattern**:
  ```
  error: call to [host] function "fmaf" from [aicore] function    // compile-time rejection
  ld.lld: error: undefined symbol: fmaf                            // link-time, for __builtin_fmaf
  ```
  Same for `__builtin_fmaf` (resolves to the `fmaf` libm name → undefined symbol at link). Distinct from CAND-A5OPS-0004's transcendental family — `fmaf` is IEEE-754 fused-multiply-add, not a trig/exp/log math function.
- **Root cause**: `fmaf` is a libc/libm HOST function; the dav-m200 aicore device runtime does not provide it, and there is no public scalar `Fma()` in the §9.1 math library that yields single-rounding FMA in S-pipe code. Critically, dav-m200 (arch200) has NO SIMT, so the OL-242 `__simt_callee__` scalar escape hatch (which carries scalar transcendentals on arch35) is ABSENT here — there is no scalar-FMA route at all on this SoC.
- **Consequence (MEASURED, not a waiver)**: without FMA, an accumulation `acc += a*b` must use separate multiply + add (two roundings) instead of one fused rounding. This produces ~1-ULP fp16 drift vs a reference computed with hardware FMA. This is a precision-target fact, not a tolerance loosening — if bit-exactness vs a hardware-FMA reference is REQUIRED, a from-scratch scalar mul+add kernel cannot meet it; declare the target honestly (T1 golden-truth PASS-with-tolerance) rather than chasing a bit-match that the rounding path precludes.
- **Detection**: compile error "host function … from aicore function" OR linker `undefined symbol: fmaf` after referencing `fmaf` / `__builtin_fmaf` → remove the FMA call; switch to `+= a*b`.
- **Evidence**: incre_flash_attention (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — three build iters rejected `fmaf` / `__builtin_fmaf` (failures-ledger iters 1/3). With separate mul+add + polynomial `FastExp`, the scalar kernel lands at `max_abs ≤ 0.000977` (1-ULP fp16) on all 27 cases vs the 910B2C golden truth; 13/27 then fail strict MARE because near-zero output elements turn the 1-ULP absolute gap into high relative error. CANN `aclnnIncreFlashAttention` is BIT-EXACT across 910B2C↔310P3 (`mere=0`, `mare=0` for cases 0–4), confirming CANN's exp+FMA path is deterministic and hardware-portable — so the residual is the exp-implementation + no-FMA rounding path, NOT algorithm divergence (`verification.json.precision.method`).
- **Fix / options**: (1) `+= a*b` accumulation, accept the measured 1-ULP, document it (do NOT mask as "expected"); (2) for `exp` specifically, route through a SIMD `Exp()` high-level-API pass per CAND-A5OPS-0004's SIMT+SIMD split IF the op tolerates a vector pass — but an all-scalar UB-free dav-m200 kernel (CAND-DAV-M200-UB-FREE-SCALAR-KERNEL) has no in-kernel FMA route; (3) if bit-exact vs CANN is mandatory, match CANN's exact exp/FMA bit-path — out of scope for a from-scratch scalar kernel, so declare FAIL-to-bit-match honestly rather than waive.
- **Cross-ref**: CAND-A5OPS-0004 (transcendentals don't link — sibling root, different math class; its dav-m200 note is why the [`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md) SIMT escape hatch is absent here), [`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md) (SIMT scalar transcendentals — arch35 only, N/A on dav-m200), CAND-DAV-M200-UB-FREE-SCALAR-KERNEL (the all-scalar architecture this constraint shaped). backend=ascendc.

**孪生卡 ec-269 合并的独有内容**（ec-269-fmaf-builtin-fmaf-is-host-only-on-dav-m200-aicor.md 已删除）：

> error: call to [host] function "fmaf" from [aicore] function // compile-time rejection

> ld.lld: error: undefined symbol: fmaf // link-time, for __builtin_fmaf

> Same for `__builtin_fmaf` (resolves to the `fmaf` libm name → undefined symbol at link). Distinct from EC-35's transcendental family — `fmaf` is IEEE-754 fused-multiply-add, not a trig/exp/log math function.

> - **Fix / options**: (1) `+= a*b` accumulation, accept the measured 1-ULP, document it (do NOT mask as "expected"); (2) for `exp` specifically, route through a SIMD `Exp()` high-level-API pass per EC-35's SIMT+SIMD split IF the op tolerates a vector pass — but an all-scalar UB-free dav-m200 kernel (CAND-DAV-M200-UB-FREE-SCALAR-KERNEL) has no in-kernel FMA route; (3) if bit-exact vs CANN is mandatory, match CANN's exact exp/FMA bit-path — out of scope for a from-scratch scalar kernel, so declare FAIL-to-bit-match honestly rather than waive.

> - **Cross-ref**: [`EC-35`](ec_35_aiv_device_binary_does_not_link_libm_transcendenta.md) (transcendentals don't link — sibling root, different math class; its dav-m200 note is why the [`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md) SIMT escape hatch is absent here), [`OL-242`](ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md) (SIMT scalar transcendentals — arch35 only, N/A on dav-m200), CAND-DAV-M200-UB-FREE-SCALAR-KERNEL (the all-scalar architecture this constraint shaped). backend=ascendc.
