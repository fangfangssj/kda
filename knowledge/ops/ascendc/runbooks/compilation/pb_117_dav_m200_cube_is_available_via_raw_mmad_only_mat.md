---
type: Runbook
title: "dav-m200 cube is available via raw Mmad — only MatmulImpl<>/Fixpipe/LoadData2D are arch-gated (2026-08-13)."
description: "dav-m200 的 cube 可经裸 Mmad 使用——只有 MatmulImpl/Fixpipe/LoadData2D 被 arch 门控；Mmad+LoadData3DParamsV2Pro 实测可用"
tags: [task.debugging, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-117"]
---

# dav-m200 cube is available via raw Mmad — only MatmulImpl<>/Fixpipe/LoadData2D are arch-gated (2026-08-13).

**PB-117** — **dav-m200 cube is available via raw Mmad — only MatmulImpl<>/Fixpipe/LoadData2D are arch-gated (2026-08-13).** The `Mmad`/`LoadData3DParamsV2Pro` NOT_SUPPORT stubs are `#if defined(__DAV_M200_VEC__)`-guarded and do NOT fire on the real 310P3 chip (`__DAV_M200__` only) → real `mad()`/`img2colv2_cbuf_to_ca` execute. Runtime probe bit-exact 20/20 seeds (GM→UB→L1→L0A/L0B→Mmad→L0C→UB→GM). Genuinely NOT_SUPPORT: `MatmulImpl<>` (arch-gated 2201/3510), Fixpipe (unconditional), `LoadData2DParamsV2`/`LoadDataWithTranspose`/GM→L0A. Gotcha: `DataCopyParams.blockLen` is in 32B datablocks NOT elements (256→8KB overflow). backend=all. 2026-08-13. Cross-ref [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md), [`P-P203`](../performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md).

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200, `__NPU_ARCH__=2002`); cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — runtime Mmad probe, 20/20 seeds bit-exact`
`unverified_on: soc=Ascend950PR, Ascend910 (cube API fully functional there — this entry is dav-m200-specific)`

Matmul on dav-m200 IS feasible — write it with raw `Mmad` + `LoadData3DParamsV2Pro` + `BLOCK_MODE_MATRIX` readout, NOT the high-level `MatmulImpl<>`. The `ASCENDC_REPORT_NOT_SUPPORT(false, "Mmad")` and `LoadData3DParamsV2Pro` stubs in `kernel_operator_mm_impl.h` are wrapped in `#if defined(__DAV_M200_VEC__)`, which is NOT defined on the real chip (`__DAV_M200__` only) — the real `mad()` / `img2colv2_cbuf_to_ca/cb` bodies execute. A 16×16×16 fp16 Mmad runtime probe (official 310P pipeline: GM→UB→L1(A1/B1)→L0A/L0B→Mmad→L0C→UB→GM) is **bit-exact** vs torch fp32 matmul across 20 seeds + identity edge cases (A×I==A, I×I==I).

**What WORKS on dav-m200 (verified at runtime)**:
| Primitive | Status |
|-----------|--------|
| `Mmad(dst, srcA, srcB, MmadParams(m,n,k,0,0,1))` | ✅ real `mad()` executes |
| `LoadData3DParamsV2Pro` (A1→L0A / B1→L0B, via `img2colv2_cbuf_to_ca/cb`) + `Load3DSetFMatrixCal(1,16,padList)` | ✅ real path |
| L0C→UB readout: `DataCopy(ub, cL0, {blockLen=1, blockCount=N}, BLOCK_MODE_MATRIX)` | ✅ fp32: blockCount=2 (two 8×16 sub-fractals); fp16: blockCount=1 |
| VEC fp32 `Muls`/`Adds` | ✅ (see PB-118) |

**What is GENUINELY NOT_SUPPORT on dav-m200** (unconditional stubs or arch-gated — do NOT confuse with the `__DAV_M200_VEC__`-guarded ones):
| API | Reason |
|-----|--------|
| `MatmulImpl<>` high-level template | `#if __NPU_ARCH__==2201 \|\| ==3510` — 2002 not in guard → use raw Mmad instead |
| `Fixpipe` all variants (SetFixPipeConfig/Addr/ClipRelu, CO1→GM/C1) | unconditional NOT_SUPPORT |
| `LoadData2DParamsV2`, `LoadDataWithTranspose`, `LoadData2D GM→L0A/B`, `LoadData2D L1→L0A/B` | unconditional NOT_SUPPORT → use LoadData3DParamsV2Pro |
| `LoadData3DParamsV2` (non-Pro) A1/B1→L0 | unconditional NOT_SUPPORT → use Pro variant |
| `LoadData3DParamsV2Pro` B1→B2 with int8_t/uint8_t | NOT_SUPPORT (half path works) |
| `LoadData3DParamsV2Pro` L1→UB | NOT_SUPPORT |
| `Mmad`+biasLocal / `MmadWithSparse` / `LoadDataWithSparse` / `BroadCastVecToMM` | unconditional NOT_SUPPORT |

**Impact**: matmul/convolution-style kernels ARE feasible on dav-m200 — write them with raw `Mmad` + `LoadData3DParamsV2Pro` + `BLOCK_MODE_MATRIX` readout instead of the high-level `MatmulImpl`. Do NOT conclude "infeasible" from a `NOT_SUPPORT` stub string — check the guard (`__DAV_M200_VEC__` vs `__DAV_M200__`) and probe at runtime.

**DataCopyParams gotcha (arch-general)**: `DataCopyParams.blockLen` for GM↔UB is in 32-byte datablocks, NOT elements. `DataCopy(dst, src, DataCopyParams(256,1,0,0))` transfers 256×32B = **8KB**, not 256 elements — overflowing into adjacent GM allocations (observed zeroing a neighboring torch tensor). For 256 half (512B) use `blockLen=16`; for 256 fp32 (1KB) use `blockLen=32`.

**Evidence**: runtime Mmad probe (2026-08-13, `/tmp/cube_probe_ws/`, 310P3 device 1, CANN 9.0.0) — 20/20 seeds `torch.equal(out, ref)` bit-exact, A×I==A, I×I==I, no memory corruption after DataCopyParams fix. grouped_matmul redo on this cube path passes 35/35 (see PB-119).

**Cross-ref**: [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md) (fp32 VEC arithmetic — same DataCopyParams gotcha class), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md) (Fixpipe L0C→UB — genuinely different value semantics on the CO1 path), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (DataCopyPad absent on 200x), [`P-P203`](../performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md) (dav-m200 VEC cast support matrix).

**孪生卡 pb-170 合并的独有内容**（pb-170-dav-m200-cube-is-available-via-raw-mmad-matmulim.md 已删除）：

> **Cross-ref**: [`PB-118`](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md) (fp32 VEC arithmetic — same DataCopyParams gotcha class), [`PB-107`](pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md) (Fixpipe L0C→UB — genuinely different value semantics on the CO1 path), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (DataCopyPad absent on 200x), P-P125 (dav-m200 VEC cast support matrix).
