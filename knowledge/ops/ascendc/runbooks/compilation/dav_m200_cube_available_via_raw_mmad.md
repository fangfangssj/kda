---
type: Runbook
title: "dav-m200 cube is available via raw `Mmad` — only `MatmulImpl<>`/`Fixpipe`/`LoadData2D` are arch-gated"
description: "dav-m200 cube is available via raw `Mmad` — only `MatmulImpl<>`/`Fixpipe`/`LoadData2D` are arch-gated"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["Fixpipe", "LoadData3DParamsV2", "BroadCastVecToMM"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-117 — PB-117: A5Ops additive extension

> 关联：同一 PB-117 主题已由既有卡 [pb_117_dav_m200_cube_is_available_via_raw_mmad_only_mat.md](pb_117_dav_m200_cube_is_available_via_raw_mmad_only_mat.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__NPU_ARCH__=2002）；CANN 9.0.0；op_class=all。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0 — 运行时 Mmad 探针，20/20 种子 bit-exact。

**PB-117** — **dav-m200 cube is available via raw Mmad — only MatmulImpl<>/Fixpipe/LoadData2D are arch-gated (2026-08-13).** The `Mmad`/`LoadData3DParamsV2Pro` NOT_SUPPORT stubs are `#if defined(__DAV_M200_VEC__)`-guarded and do NOT fire on the real 310P3 chip (`__DAV_M200__` only) → real `mad()`/`img2colv2_cbuf_to_ca` execute. Runtime probe bit-exact 20/20 seeds (GM→UB→L1→L0A/L0B→Mmad→L0C→UB→GM). Genuinely NOT_SUPPORT: `MatmulImpl<>` (arch-gated 2201/3510), Fixpipe (unconditional), `LoadData2DParamsV2`/`LoadDataWithTranspose`/GM→L0A. Gotcha: `DataCopyParams.blockLen` is in 32B datablocks NOT elements (256→8KB overflow). backend=all. 2026-08-13. Cross-ref [PB-118](dav_m200_vec_fp32_arithmetic_works.md), [PB-107](dav_m200_fixpipe_fp32_to_fp16_saturates_not_inf.md), [PB-105](dav_m200_datacopypad_unsupported_both_directions.md), P-P203（A5Ops 绑定外部引用；本语料未收录目标）.

- Former A5Ops ID: `PB-117`
- Fusion disposition: `additive_new`

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

**Cross-ref**: PB-118 (fp32 VEC arithmetic — same DataCopyParams gotcha class), PB-107 (Fixpipe L0C→UB — genuinely different value semantics on the CO1 path), PB-105 (DataCopyPad absent on 200x), P-P203 (dav-m200 VEC cast support matrix).
