---
type: Runbook
title: "predicate/mask cast chain on dav-m200"
description: "dav-m200 的 vconv 支持矩阵刚好凑齐 f32→f16→u8 谓词转换链，220x 的 int16-lane 谓词算法可原样照搬；int16→u8 直连不支持，写码前先查 SDK 头文件"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P203"]
---

# predicate/mask cast chain on dav-m200

## 优化点

**P-P203** — **predicate/mask cast chain on dav-m200** (promoted from CAND-DAV-M200-VCONV-PREDICATE-CAST-SUPPORT, 2 independent ops): port int16-lane bit-predicate algorithms via the f32→f16(CAST_NONE)→u8(CAST_CEIL) half-roundtrip (+ s32→f32 CAST_NONE); direct int16→u8 narrowing is `ASCENDC_REPORT_NOT_SUPPORT`; verify the per-SoC vconv matrix at SDK header level pre-build ([`OL-130`](../compilation/ol_130_v220_libascendcl_soname_conflict.md)). is_finite 310p (42/42 bit-exact, first-build PASS) + is_inf 310p 2026-08-09 (220x CompareInf verbatim, 42/42, det 42/42, perf 1.29×/1.27× device-event). Cross-ref [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md), EC-88, [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md)/[`OL-130`](../compilation/ol_130_v220_libascendcl_soname_conflict.md), [`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md), [`OL-404`](../compilation/ol_404_a_captured_ground_truth_speaks_only_for_the_chip.md).

(promoted from CAND-DAV-M200-VCONV-PREDICATE-CAST-SUPPORT, 2026-08-09 — 2 independent ops)

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=elementwise predicate / mask-producing kernels`
`verified_on: soc=Ascend310P3; cann=9.0.0 (is_finite 310p + is_inf 310p, both 2026-08-09)`
`unverified_on: soc=Ascend310B, Ascend910, Ascend950PR (vconv support matrices are per-SoC SDK codegen — re-read the target's own kernel_operator_vec_vconv_impl.h before relying)`

**Principle**: for predicate/mask algorithms that lane-pack via an int16 half-roundtrip
(float→half, then half→uint8 mask byte), the dav-m200 SDK vconv matrix provides exactly the
casts needed — `vconv_s322f32` (s32→f32, CAST_NONE), `vconv_f322f16` (f32→f16, CAST_NONE),
`vconv_f162u8c` (f16→u8, CAST_CEIL; FLOOR/ROUND/TRUNC/NONE also present) — so an upstream
220x predicate algorithm ports 1:1 with ZERO compute changes. The trap is the DIRECT
narrowing cast: int16→uint8 is `ASCENDC_REPORT_NOT_SUPPORT`, half→int16 exists only as
CAST_RINT, int16→half only as CAST_NONE. Verify cast support at the SDK header level BEFORE
coding the kernel, not at compile time (OL-130's lookup discipline).

**Concrete anchor**: CANN 安装树内 cann-9.0.0/aarch64-linux/asc/impl/basic_api/dav_m200/kernel_operator_vec_vconv_impl.h:175-215,307-345
(the dav-m200 vconv support matrix; grep-able, per-SoC). Vendor-faithfulness note: the
upstream fp32 path's per-int16-lane `& 0x7fff` after the int32 subtract carries a quirk
(NaN ±0x7f808000, mantissa==0x00008000 exactly, misclassified as inf) — port it verbatim;
it is absent from quiet-NaN (0x7fc00000) captures.

**Evidence**:
- is_finite 310p (2026-08-09, first instance): upstream int16-lane is_finite algorithm
  (fp32/fp16 → bool predicate) ported with zero compute changes; build PASS first iteration,
  42/42 bit-exact vs A3 capture, det 42/42.
- is_inf 310p (2026-08-09, promoting instance): upstream 220x CompareInf ported verbatim
  onto the is_finite dav-m200 structural skeleton (P-P201 in-kernel tail; PB-105/PB-106
  avoided); all required casts confirmed at SDK header level pre-build; 1 build iter + 1
  precision iter, both first-try PASS; 42/42 bit-exact vs Path-A capture, det 42/42; perf
  1.29×/1.27× device-event across two independent runs (P-P202 menu applied at kw-1).

**Other instances (predicted)**: is_nan, sign-mask, comparison-mask, and any mask-producing
elementwise op ported to dav-m200 whose 220x algorithm computes predicates in int16 lanes.

**Cross-ref**: [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md) (dav-m200 missing-API class — DataCopyPad), EC-88 (arch-220x
missing-cast twin — `Cast<float,int8_t>` two-step workaround), OL-80/OL-130 (API catalog +
SDK-header lookup chain), P-P201 (the sibling skeleton this ports onto), OL-404 (audit the
capture before relying on any NaN-semantics advisory). backend=ascendc.

## 适用场景

- **工程形态**：dav-m200 的 vconv 支持矩阵刚好凑齐 f32→f16→u8 谓词转换链。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
