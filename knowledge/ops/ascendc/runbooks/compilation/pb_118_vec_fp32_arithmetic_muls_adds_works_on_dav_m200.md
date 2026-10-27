---
type: Runbook
title: "VEC fp32 arithmetic (Muls/Adds) works on dav-m200 (2026-08-13)."
description: "dav-m200 的 VEC fp32 算术（Muls/Adds）实测正常——按位与参考一致，可以放心用"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-118"]
---

# VEC fp32 arithmetic (Muls/Adds) works on dav-m200 (2026-08-13).

**PB-118** — **VEC fp32 arithmetic (Muls/Adds) works on dav-m200 (2026-08-13).** Probe: `Muls(*,2.0f)`, `Adds(*,1.5f)`, identity-`Adds(*,0.0f)` on 256 fp32 all `max_err=0.0`. dav_m200 `Adds` supports float (`SupportType<T, half, float, int16_t, int32_t>`); official 310P CANN kernel runs fp32 `Muls`. `Duplicate(dst, src_tensor, count)` has no fp32 form (scalar-fill only). If fp32 output looks zero/garbage, check GM↔UB DataCopyParams units before blaming hardware. GlobalTensor half uses element offset; SetGlobalBuffer bufferSize is element count. ⚠ `Cast<float,half>` ✅ row is CAST_NONE-scoped — see [`PB-119`](pb_119_dav_m200_fp32_fp16_cast_supports_only_cast_none.md). backend=all. 2026-08-13. Cross-ref [`PB-117`](pb_117_dav_m200_cube_is_available_via_raw_mmad_only_mat.md), CAND-M200-VEC-MULS-SUBNORMAL-NONIEEE, [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md), [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md).

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

**Cross-ref**: [`PB-117`](pb_117_dav_m200_cube_is_available_via_raw_mmad_only_mat.md) (same DataCopyParams gotcha class), CAND-M200-VEC-MULS-SUBNORMAL-NONIEEE (subnormal-specific Muls behavior — distinct), [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (fp32 DataCopy GM→UB 32-byte write limit — different pipe, different gap class), [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (fp16 Cast RoundMode — **see [`PB-119`](pb_119_dav_m200_fp32_fp16_cast_supports_only_cast_none.md) for the dav-m200 CAST_NONE-only caveat**).

**孪生卡 pb-171 合并的独有内容**（pb-171-vec-fp32-arithmetic-muls-adds-works-on-dav-m200.md 已删除）：

> | `Cast<float,float>` (identity-width) | ❌ runtime NOT_SUPPORTED (see PB-187) — use the `Adds(dst, src, 0.0f)` identity row above for fp32 TQue↔TBuf staging |

## 适用场景

- **工程形态**：dav-m200 的 VEC fp32 算术（Muls/Adds）实测正常。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
