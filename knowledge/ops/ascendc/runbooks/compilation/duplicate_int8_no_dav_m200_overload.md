---
type: Runbook
title: "`Duplicate<int8_t>` has no dav-m200 overload — reinterpret the buffer via `Get<int16_t>()` + `Duplicate<int16_t>`"
description: "`Duplicate<int8_t>` has no dav-m200 overload — reinterpret the buffer via `Get<int16_t>()` + `Duplicate<int16_t>`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-191 — EC-191: A5Ops additive extension

> 关联：同一 EC-191 主题已由既有卡 [ec_191_duplicate_int8_t_has_no_dav_m200_overload.md](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；bisheng=dav-m200 toolchain；op_class=all（任何对 8-bit buffer 做清零/广播填充的 kernel）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（diag_v2 310p，2026-08-09，worker iter 1 编译期捕获；修复后 8/8 bit-exact T1）。

**EC-191** — **`Duplicate<int8_t>` has no dav-m200 overload** (compile-time `SupportType` static_assert: half/i16/u16/i32/u32/f32 only); fix = reinterpret the buffer via `Get<int16_t>()` + `Duplicate<int16_t>` — the vendor's own idiom (ops-math 仓（公开）/.../diag_flat_nd_to_2d.h:279,327 always Duplicates through an int16 view; copy the idiom, not the dtype). diag_v2 310p 2026-08-09 (iter-1 build fail → 8/8 bit-exact T1). Cross-ref [EC-88](cast_float_int8_no_arch_220x_overload.md), [PB-105](dav_m200_datacopypad_unsupported_both_directions.md), P-P203（A5Ops 绑定外部引用；本语料未收录目标）, OL-405（A5Ops 绑定外部引用；本语料未收录目标）/[OL-130](ol_130_v220_libascendcl_soname_conflict.md).

- Former A5Ops ID: `EC-191`
- Fusion disposition: `additive_new`

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
  (`workspace/diag_v2/failures_ledger.md` iter 1; workspace scratch, gitignored).
- **Cross-ref**: EC-88 (the Cast float←int8 no-overload sibling on 220x), PB-105
  (DataCopyPad absent on dav-m200 — same SDK-capability-gap class, data-movement twin),
  P-P203 (dav-m200 cast/predicate support matrix — verify at SDK header level pre-build),
  OL-405, OL-130. backend=ascendc.
