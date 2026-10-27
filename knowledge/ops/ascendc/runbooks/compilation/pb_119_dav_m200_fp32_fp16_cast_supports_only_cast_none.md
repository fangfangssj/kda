---
type: Runbook
title: "dav-m200 fp32→fp16 Cast supports ONLY CAST_NONE/CAST_ODD — every rounding mode produces garbage/assert (VERIFI"
description: "dav-m200 fp32→fp16 Cast 只支持 CAST_NONE/CAST_ODD，其他舍入模式产垃圾或断言（已实证）"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-119"]
---

# dav-m200 fp32→fp16 Cast supports ONLY CAST_NONE/CAST_ODD — every rounding mode produces garbage/assert (VERIFI

**PB-119** — **dav-m200 fp32→fp16 `Cast` supports ONLY `CAST_NONE`/`CAST_ODD` — every rounding mode produces garbage/assert (VERIFIED 2026-08-13).** `CAST_RINT` garbage in ALL contexts (pure-VEC probe maxdiff 96.4; cube-kernel too); `CAST_ROUND`/`CAST_FLOOR` bit-exact in pure-VEC but garbage in cube-kernel context (maxdiff 538); non-NONE/ODD modes fire `ASCENDC_ASSERT(false)` in `vconv` impl. `CAST_NONE` (`vconv_f322f16`) bit-exact everywhere. Root cause of grouped_matmul Mmad-cube redo all-NaN → fixed with CAST_NONE → 35/35 PASS vs CPU fp64 + 35/35 vs A3 CANN-truth. **Rule: on dav-m200, fp32→fp16 Cast MUST use CAST_NONE.** [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md)'s blanket CAST_RINT default is dav-m200-scoped by this entry. backend=all. verified 2026-08-13 (310P3 device 1, CANN 9.0.0). Cross-ref [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (CAST_RINT default is dav-m200-scoped), [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md).

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200, arch 2002); cann=9.0.0; op_class=any kernel with an fp32→fp16 output Cast`
`verified_on: soc=Ascend310P3 (dav-m200, device 1); cann=9.0.0 — runtime cast probe + cube-kernel probe + grouped_matmul 35/35 PASS`
`unverified_on: soc=Ascend950PR, Ascend910 (arch35/351x & 220x support the rounding modes — this is dav-m200-specific)`

**PROVEN (2026-08-13)**: `Cast(LocalTensor<half>, LocalTensor<float>, RoundMode, count)` (fp32→fp16) on dav-m200:
- `CAST_NONE` (`vconv_f322f16`) — **WORKS, bit-exact everywhere** (pure-VEC AND cube-kernel context).
- `CAST_ODD` (`vconv_f322f16o`) — works (per header + upstream 220x mirror).
- `CAST_RINT` — **broken in ALL contexts** (pure-VEC probe: garbage, maxdiff 96.4; cube-kernel: garbage). ⚠ This directly CONTRADICTS OL-81's blanket "use CAST_RINT for fp16" default when porting to dav-m200.
- `CAST_ROUND` / `CAST_FLOOR` — bit-exact in a **pure-VEC** kernel, but **garbage in a cube-kernel context** (mixed probe2: maxdiff 538, 0% match). Consequence: you cannot rely on any rounding mode inside a kernel that also runs Mmad/LoadData — only CAST_NONE is safe.
- Manifestation is **silent garbage / NaN, NOT a clean error** — `CastIntrinsicsImpl` in `dav_m200/kernel_operator_vec_vconv_impl.h` (~lines 210–245) fires `ASCENDC_ASSERT(false)` for non-NONE/ODD modes, and at runtime the destination is left garbage/uninitialized (observed wrong finite values + NaN).

**Why this is a trap**: OL-81's "CAST_RINT not CAST_ROUND (IEEE RNE)" guidance is correct for arch35/351x and 220x, but WRONG for dav_m200/2002 float→half. On dav-m200 the hardware-native path is `vconv_f322f16` = CAST_NONE. Upstream CANN kernels confirm this: `#if __CCE_AICORE__ == 200` branches select CAST_NONE for fp32→fp16 output casts (pows, mse_loss_v2, ge_glu_grad_v2 — all previously noted in OL-81 as "upstream-fidelity" choices, now explained: the hardware REJECTS the rounding modes).

**Impact**: grouped_matmul Mmad-cube redo was ALL-NaN because the epilogue used `CAST_RINT`; switching to `CAST_NONE` gave **35/35 PASS** vs CPU fp64 AND A3 CANN-truth (K=512 cases 0.67/0.76 → 0.031). Every dav-m200 kernel with an fp32→fp16 output cast MUST use CAST_NONE.

**Rule for dav-m200**: fp32→fp16 Cast → `RoundMode::CAST_NONE` (or CAST_ODD). Never CAST_RINT/ROUND/FLOOR/CEIL/TRUNC. For fp16→fp32 Cast, CAST_NONE works (asymmetric support verified: half→fp32 CAST_NONE bit-exact).

**Evidence**: (1) cast probe `/tmp/cube_probe_ws/kernel/cast_probe_kernels.cpp` (pure-VEC, 4 round modes, CAST_RINT garbage vs NONE/ROUND/FLOOR bit-exact); (2) mixed probe2 `/tmp/cube_probe_ws/kernel/mixed_probe2_round_kernels.cpp` (cube-kernel context: raw A@B fp32 bit-exact, A@B+bias fp32 bit-exact, fp16 cast garbage); (3) grouped_matmul worker 3-kernel isolation (cube-only bit-exact, add-only bit-exact, cast-only garbage) + 35/35 PASS on CAST_NONE. All on 310P3 device 1, CANN 9.0.0, 2026-08-13.

**Cross-ref**: [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (the CAST_RINT default — dav-m200-scoped by this entry), [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md) (`Cast<float,half>` ✅ row requires CAST_NONE), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (older dav-m200 refutation chain), CAND-M200-... (dav-m200 platform catalog).
**Cross-ref**: [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (the 220x 32B-alignment sibling — same width class, different symptom/SoC), [`PB-106`](pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (the host-staging race this mitigation does NOT trigger, with the stream-ordering distinction), OL-181 (host-side allocation padding for the 220x elementwise-alignment sibling). backend=ascendc.
