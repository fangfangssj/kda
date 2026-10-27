---
type: Runbook
title: "Duplicate<int8_t> has no dav-m200 overload"
description: "`Duplicate<int8_t>` 在 dav-m200 没有重载（白名单仅 half/i16/u16/i32/u32/f32）；解法是把 buffer reinterpret 成 int16 视图再 Duplicate——厂商自己的惯用法。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-191"]
---

# Duplicate<int8_t> has no dav-m200 overload

**EC-191** — **`Duplicate<int8_t>` has no dav-m200 overload** (compile-time `SupportType` static_assert: half/i16/u16/i32/u32/f32 only); fix = reinterpret the buffer via `Get<int16_t>()` + `Duplicate<int16_t>` — the vendor's own idiom (ops-math 仓（公开）/.../diag_flat_nd_to_2d.h:279,327 always Duplicates through an int16 view; copy the idiom, not the dtype). diag_v2 310p 2026-08-09 (iter-1 build fail → 8/8 bit-exact T1). Cross-ref EC-88, [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md), [`P-P203`](../performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md), [`OL-405`](../performance/ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md)/[`OL-130`](ol_130_v220_libascendcl_soname_conflict.md).

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=dav-m200 toolchain; op_class=all (any kernel zero-filling / broadcast-filling an 8-bit buffer)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (diag_v2 310p, 2026-08-09, compile-caught at worker iter 1)`

- **Symptom**: `Duplicate<int8_t>(...)` fails at COMPILE time — the dav-m200 `SupportType`
  static_assert accepts only half / int16 / uint16 / int32 / uint32 / fp32. Same
  missing-dtype-pair class as EC-88 (`Cast<float,int8_t>` has no arch-220x overload), one
  primitive over. Other 8-bit dtypes (uint8) are outside the same list — expected identical,
  unverified.
- **Fix — reinterpret to a supported width**: view the int8 buffer as int16
  (`buf.Get<int16_t>()`) and `Duplicate<int16_t>` through the view (element count halves;
  handle an odd tail by scalar `SetValue` or a 32B-aligned pad). This is not an invented
  workaround: upstream ops-math 仓（公开）/conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h:279,327
  ALWAYS Duplicates through an int16 view — the reinterpret idiom exists in the source
  precisely because the 200x VEC unit lacks 8-bit Duplicate. **Copy the idiom, not the
  dtype** (OL-405: the source's own 200x-compatibility choices are the design reference).
- **Detection**: any `Duplicate<int8_t>` / 8-bit `Duplicate` in a 200x-target TU → expect
  the static_assert. Read the SDK `SupportType` list for the TARGET arch before choosing a
  fill dtype (OL-130 header-lookup chain); do NOT shop dtypes by trial compile.
- **Evidence**: diag_v2 310p (2026-08-09, first instance) — iter-1 build failure on
  `Duplicate<int8_t>` zero-fill of the assist buffers; fixed via `Get<int16_t>()` view +
  `Duplicate<int16_t>`; then 8/8 bit-exact T1, det 8/8
  (`diag_v2/failures_ledger.md` iter 1; workspace scratch, gitignored).
- **Cross-ref**: EC-88 (the Cast float←int8 no-overload sibling on 220x), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md)
  (DataCopyPad absent on dav-m200 — same SDK-capability-gap class, data-movement twin),
  P-P203 (dav-m200 cast/predicate support matrix — verify at SDK header level pre-build),
  OL-405, OL-130. backend=ascendc.

## 适用场景

- **工程形态**：`Duplicate<int8_t>` 在 dav-m200 没有重载（白名单仅 half/i16/u16/i32/u32/f32）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
