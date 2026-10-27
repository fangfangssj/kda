---
type: Runbook
title: "`floorf`/`floor` unavailable in the `__aicore__` kernel TU on dav-m200 — replace with a manual integer floor"
description: "`floorf`/`floor` unavailable in the `__aicore__` kernel TU on dav-m200 — replace with a manual integer floor"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["error: use of undeclared identifier 'floorf'"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-196 — EC-196: A5Ops additive extension

> 关联：同一 EC-196 主题已由既有卡 [ec_196_floorf_floor_unavailable_in_the_aicore_kernel_tu.md](ec_196_floorf_floor_unavailable_in_the_aicore_kernel_tu.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3；CANN 9.0.0；op_class=all。
- 验证来源：soc=Ascend310P3（dav-m200，arch 200x）；CANN 9.0.0（multi_scale_deformable_attn_function 310p，2026-08-10；手工 floor 后 9/9 T1 PASS）。

**EC-196** — `floorf`/`floor` unavailable in the `__aicore__` kernel TU on dav-m200 (ccec compile: "undeclared identifier 'floorf'"); replace with a manual integer floor (`int32_t i = static_cast<int32_t>(x); if ((float)i > x) i--;`). Truncation toward zero is WRONG for negative coords (grid-sample / `align_corners=False` `pixel = loc*size - 0.5` is frequently negative). DISTINCT from CAND-A5OPS-0004 (libm TRANSCENDENTALS absent at LINK, fixed by SIMT/SIMD split) — floor is non-transcendental, fix is a 2-line manual floor, no extra launch. COMPILE claim (build-host), device provenance UNVERIFIED — does not block it. Anchor: 内部路径（不公开）workspace/multi_scale_deformable_attn_function/kernel/multi_scale_deformable_attn_function_kernel.h:167-168, 9/9 T1. Cross-ref CAND-A5OPS-0004, [EC-193](aicore_unsigned_to_float_static_cast_rejected.md). backend=ascendc.

- Former A5Ops ID: `EC-196`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the libm-symbol absence is arch-family-wide per CAND-A5OPS-0004; the manual-floor fix is portable everywhere, so promotion to soc=all is expected once a 2nd SoC confirms`

- **Category**: compile / authoring (API-surface)
- **Loaded by**: Generator/Worker when a kernel needs `floor`/`floorf` (coordinate math, grid-sample / bilinear-interpolation pixel coords, tiling divmod, index quantization)
- **Error pattern**: at the ccec compile stage the kernel TU reports `floorf` (or `floor`) as undefined in the CCE kernel environment (`error: use of undeclared identifier 'floorf'`). This is the basic-math (`<cmath>`) floor symbol — NOT a transcendental. It is distinct from CAND-A5OPS-0004, which is a LINK-stage libm absence for transcendentals (`cosf`/`sinf`/`expf`/...) fixed by a SIMT→SIMD split; here the function is non-transcendental and the fix is a 2-line manual integer floor, not an extra kernel launch.
- **Root cause**: the `__aicore__` device TU is not linked against libm and does not intrinsic-provide `floorf`; the symbol is simply unavailable to kernel code. Reach for it in any coordinate/indexing math and the compile rejects it.
- **Fix**: implement floor-toward-negative-infinity by hand. Naive `static_cast<int32_t>(x)` truncates toward ZERO and is wrong for negative coordinates — exactly the grid-sample / `align_corners=False` case where `pixel = loc*size - 0.5` is frequently negative:
  ```cpp
  // floorf() not available in the AscendC kernel environment
  int32_t x0 = static_cast<int32_t>(x_pix);   // truncates toward zero
  if ((float)x0 > x_pix) x0--;                // correct toward -inf for negatives
  ```
- **Detection**: grep the kernel TU for `floorf` / `floor(` / `std::floor`; any hit in device code → replace with the manual form above. Most common in interpolation / grid-sample / RoI-align pixel coords and in any divmod needing the mathematical floor of a possibly-negative quotient.
- **Evidence**: multi_scale_deformable_attn_function 310p port (2026-08-10, Ascend310P3/dav-m200, arch 200x, CANN 9.0.0, port_a3_to_a5) — 内部路径（不公开）workspace/multi_scale_deformable_attn_function/kernel/multi_scale_deformable_attn_function_kernel.h:167-168 hit "floorf not available in AscendC kernel environment" at compile (logged `novel` in 内部路径（不公开）workspace/multi_scale_deformable_attn_function/failures_ledger.md:4 iter 1); replaced with the truncation + negative-correction form. Bilinear interpolation (`align_corners=False`, `pixel = loc*size - 0.5`) then matched CPU-canonical `F.grid_sample` across 9/9 T1 PASS. API-surface / compile claim, so the spec's `device.runtime: NOT_RUN` provenance flag does not block it (symbol absence is visible at compile, independent of device execution).
- **Other instances (predicted)**: any op with sub-pixel coordinate math — grid_sample, affine_grid, RoIAlign, deformable attention, NMS coordinate rounding, tiling `floor(N/tile)` when the quotient can be negative. Where the value is known non-negative (e.g. unsigned tiling counts), bare `static_cast<int32_t>` suffices; the correction is only needed where the operand can be < 0.
- **Cross-ref**: CAND-A5OPS-0004 (libm transcendentals unavailable at LINK — different stage + function class, fixed by SIMT/SIMD split, not a manual floor); EC-193 (`__aicore__` forbids unsigned→float cast — same "device-TU restricts a plain C++ operation" family). backend=ascendc.
