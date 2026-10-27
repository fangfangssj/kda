---
type: Runbook
title: "VEC fp32 arithmetic (`Muls`/`Adds`) works on dav-m200"
description: "VEC fp32 arithmetic (`Muls`/`Adds`) works on dav-m200"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-118 — PB-118: A5Ops additive extension

> 关联：同一 PB-118 主题已由既有卡 [pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md](pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，arch 200x）；CANN 9.0.0；op_class=all（API 面行为，非编译器 codegen）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0 — 运行时 VEC 探针，bit-exact。

**PB-118** — **VEC fp32 arithmetic (Muls/Adds) works on dav-m200 (2026-08-13).** Probe: `Muls(*,2.0f)`, `Adds(*,1.5f)`, identity-`Adds(*,0.0f)` on 256 fp32 all `max_err=0.0`. dav_m200 `Adds` supports float (`SupportType<T, half, float, int16_t, int32_t>`); official 310P CANN kernel runs fp32 `Muls`. `Duplicate(dst, src_tensor, count)` has no fp32 form (scalar-fill only). If fp32 output looks zero/garbage, check GM↔UB DataCopyParams units before blaming hardware. GlobalTensor half uses element offset; SetGlobalBuffer bufferSize is element count. ⚠ `Cast<float,half>` ✅ row is CAST_NONE-scoped — see PB-119. backend=all. 2026-08-13. Cross-ref [PB-117](dav_m200_cube_available_via_raw_mmad.md), CAND-M200-VEC-MULS-SUBNORMAL-NONIEEE, [PB-22](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md), [OL-81](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md).

- Former A5Ops ID: `PB-118`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a (API-surface behavior, not compiler codegen); op_class=all`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — runtime VEC probe, bit-exact`
`unverified_on: soc=Ascend950PR, Ascend910 (fp32 VEC arithmetic functional — not dav-m200-specific)`

VEC arithmetic on fp32 operands works bit-exact on dav-m200 — no need to force fp16-only accumulation. A probe (half→`Cast`→fp32, then `Muls(*,2.0f)`, `Adds(*,1.5f)`, identity-`Adds(*,0.0f)` on 256 fp32) is **bit-exact** (`max_err = 0.0`). The dav_m200 `Adds` signature lists float in `SupportType<T, half, float, int16_t, int32_t>`; the official 310P CANN kernel (内部路径（不公开）incre_flash_attention_cube_310P_kvquant.h:1593) runs `Muls` on an fp32 LocalTensor. ⚠ `Duplicate(dst, src_tensor, count)` has NO fp32 tensor form — `Duplicate` is scalar-fill only (`Duplicate(dst, scalar, count)`).

**Working on dav-m200 (verified)**:
| Operation | Result |
|-----------|--------|
| `Cast<half,float>` / `Cast<float,half>` | ✅ bit-exact (conversion) — ⚠ fp32→fp16 must use `CAST_NONE`/`CAST_ODD`, ALL rounding modes are garbage/assert on dav-m200 (see PB-119) |
| `Muls(dst_fp32, src_fp32, scalar_fp32, count)` | ✅ bit-exact |
| `Adds(dst_fp32, src_fp32, scalar_fp32, count)` | ✅ bit-exact |
| `Adds(dst_fp32, src_fp32, 0.0f, count)` (identity/copy idiom) | ✅ bit-exact |
| fp16 arithmetic | ✅ |

**Impact on kernel design**: fp32 VEC arithmetic is available on dav-m200. If a kernel still sees "all zeros / garbage" fp32 output, suspect the GM↔UB `DataCopyParams` units (blockLen in 32B datablocks, not elements) or count/stride errors before concluding a hardware gap.

**GlobalTensor notes**:
- `GlobalTensor<half>::operator[]` uses element offset (confirmed: `address_ + offset` where `address_` is `__gm__ half*`)
- `SetGlobalBuffer(buffer, bufferSize)`: `bufferSize` is element count (not byte count)

**Evidence**: VEC fp32 probe (2026-08-13, `/tmp/cube_probe_ws/`, 310P3 device 1, CANN 9.0.0) — `Muls(*,2.0)`, `Adds(*,1.5)`, identity-`Adds` all `max_err=0.0` vs torch fp32 over 256 random fp16→fp32 values (0/256 zeros in output). grouped_matmul's fp32-capable + Mmad cube pipeline passes 35/35.

**Cross-ref**: PB-117 (same DataCopyParams gotcha class), CAND-M200-VEC-MULS-SUBNORMAL-NONIEEE (subnormal-specific Muls behavior — distinct), PB-22 (fp32 DataCopy GM→UB 32-byte write limit — different pipe, different gap class), OL-81 (fp16 Cast RoundMode — **see PB-119 for the dav-m200 CAST_NONE-only caveat**).
