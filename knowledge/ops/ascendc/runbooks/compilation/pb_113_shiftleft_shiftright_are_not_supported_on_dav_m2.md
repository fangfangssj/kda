---
type: Runbook
title: "ShiftLeft / ShiftRight are NOT SUPPORTED on dav-m200 and fail SILENTLY (ASCENDC_REPORT_NOT_SUPPORT in 内部路径（不公开）impl"
description: "dav-m200 不支持 ShiftLeft/ShiftRight（静默 NOT_SUPPORT）——用 Muls(1<<23) 之类的乘法替代移位构造 2^k"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-113"]
---

# ShiftLeft / ShiftRight are NOT SUPPORTED on dav-m200 and fail SILENTLY (ASCENDC_REPORT_NOT_SUPPORT in 内部路径（不公开）impl

**PB-113** — `ShiftLeft` / `ShiftRight` are NOT SUPPORTED on dav-m200 and fail SILENTLY (`ASCENDC_REPORT_NOT_SUPPORT` in CANN 安装树内 impl/dav_m200/kernel_operator_vec_binary_scalar_impl.h compiles clean, runs as a no-op) — a `(k+127)<<23` 2^k reconstruction produced the UN-shifted value. Fix: int `Muls(t, t, 1<<s)` (vmuls s32 supported; overflow-check

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a (SDK impl gap); op_class=all`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — on-device observation)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (ShiftLeft works on 220x/351x — the gap is dav-m200-SDK-specific; on m200, distrust ANY op whose impl is `ASCENDC_REPORT_NOT_SUPPORT`)`

- **Severity**: HIGH — the worst failure mode: compiles clean, no runtime error, the shift simply does not happen.
- **Symptom**: `ShiftLeft(dst, src, shiftCount, count)` / `ShiftRight` leave `dst` holding the UN-shifted values. Observed: a `2^k` bit reconstruction `(k+127)<<23` produced `(k+127)` unshifted (127 interpreted as fp32 bits ≈ 1.78e-43) — downstream math collapsed (in gelu: erfc≡0, output w∈{0,2}).
- **Mechanism**: the dav-m200 SDK binary-scalar impl header marks the shift path `ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftLeft")` (CANN 安装树内 impl/dav_m200/kernel_operator_vec_binary_scalar_impl.h) — on this SDK that marker compiles and runs as a SILENT no-op rather than erroring. Same trap class as the known s32→f32 Cast CAST_NONE-only quirk: an op whose dav-m200 impl is `ASCENDC_REPORT_NOT_SUPPORT` cannot be trusted to fail loudly.
- **Fix**: replace `ShiftLeft(t, t, s, count)` with an int `Muls(t, t, 1<<s, count)` — vmuls s32 IS supported (overflow check: operands must satisfy |x|·2^s ≤ INT32_MAX; for fp32 exponent construction k+127 ≤ 253, so 253·2^23 < INT32_MAX holds). For right shifts there is no exact multiply form — reformulate (Cast scaling or a different bit trick) instead of emulating.
- **Evidence**: gelu 310p (2026-08-12) — failures_ledger precision iter 1: max_abs_diff_large traced to the un-shifted `2^k` reconstruction; the int-Muls fix shipped in the same chain → 29/29 Tier-1 PASS, det 29/29 bit-identical. Kernel anchor `代码片段 gelu/gelu_kernel.h.snippet.h（见文末）` (workspace scratch, gitignored).
- **Other instances (predicted)**: any dav-m200 kernel doing fp32 bit manipulation (2^k exponent construction, mantissa extract/insert, magic-number float tricks). Note P-P200's `2^k` recipe already prefers `Adds(+127)→Muls(2^23)→Cast` for an independent reason ([`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md) int32 binary-scalar arithmetic) — that form is also the m200-safe one. Cross-ref [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-108`](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md)/[`PB-112`](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md) (same dav-m200 `ASCENDC_REPORT_NOT_SUPPORT` class), [`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md), P-P200, [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (lookup-chain extension: grep the per-SoC impl header before first use).

## 内嵌代码片段

### gelu/gelu_kernel.h.snippet.h

```cpp
            uint32_t chunks = chunksLeft > TILE_CHUNKS ? TILE_CHUNKS : static_cast<uint32_t>(chunksLeft);
            uint32_t num = chunks * CHUNK_ELEMS;
            CopyIn(offset, num);
            Compute(num);
            CopyOut(offset, num);
            offset += num;
            chunksLeft -= chunks;
        // R(z): Horner in u = min(z, ZMAX)*U_SCALE - 1 (domain-clamped so the
        // poly never extrapolates: R(ZMAX) is bounded and f below kills the tail)
        Mins(tC, tA, ZMAX_F, num);
        Muls(tC, tC, U_SCALE, num);
        Adds(tC, tC, -1.0f, num);                        // u
        Duplicate(tD, R_COEF[18], num);
        for (int k = 17; k >= 0; --k) {
```

**孪生卡 pb-166 合并的独有内容**（pb-166-shiftleft-shiftright-are-not-supported-on-dav-m2.md 已删除）：

> - **Other instances (predicted)**: any dav-m200 kernel doing fp32 bit manipulation (2^k exponent construction, mantissa extract/insert, magic-number float tricks). Note P-P122's `2^k` recipe already prefers `Adds(+127)→Muls(2^23)→Cast` for an independent reason ([`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md) int32 binary-scalar arithmetic) — that form is also the m200-safe one. Cross-ref [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-108`](pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md)/[`PB-112`](pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md) (same dav-m200 `ASCENDC_REPORT_NOT_SUPPORT` class), [`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md), P-P122, [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (lookup-chain extension: grep the per-SoC impl header before first use).

## 适用场景

- **工程形态**：dav-m200 不支持 ShiftLeft/ShiftRight（静默 NOT_SUPPORT）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
