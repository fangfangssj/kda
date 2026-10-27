---
type: Runbook
title: "floorf/floor unavailable in the __aicore__ kernel TU on dav-m200 (ccec compile: \"undeclared identifier 'floorf"
description: "dav-m200 kernel TU 没有 `floorf`；手写整数 floor（注意向零截断对负坐标是错的，grid-sample 场景高发）。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-196"]
---

# floorf/floor unavailable in the __aicore__ kernel TU on dav-m200 (ccec compile: "undeclared identifier 'floorf

**EC-196** — `floorf`/`floor` unavailable in the `__aicore__` kernel TU on dav-m200 (ccec compile: "undeclared identifier 'floorf'"); replace with a manual integer floor (`int32_t i = static_cast<int32_t>(x); if ((float)i > x) i--;`). Truncation toward zero is WRONG for negative coords (grid-sample / `align_corners=False` `pixel = loc*size - 0.5` is frequently negative). DISTINCT from CAND-A5OPS-0004 (libm TRANSCENDENTALS absent at LINK, fixed by SIMT/SIMD split) — floor is non-transcendental, fix is a 2-line manual floor, no extra launch. COMPILE claim (build-host), device provenance UNVERIFIED — does not block it. Anchor: `代码片段 multi_scale_deformable_attn_function/multi_scale_deformable_attn_function_kernel.h.snippet.h（见文末）`, 9/9 T1. Cross-ref CAND-A5OPS-0004, [`EC-193`](ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md). backend=ascendc.

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
- **Evidence**: multi_scale_deformable_attn_function 310p port (2026-08-10, Ascend310P3/dav-m200, arch 200x, CANN 9.0.0, port_a3_to_a5) — `代码片段 multi_scale_deformable_attn_function/multi_scale_deformable_attn_function_kernel.h.snippet.h（见文末）` hit "floorf not available in AscendC kernel environment" at compile (logged `novel` in `multi_scale_deformable_attn_function/failures_ledger.md:4` iter 1); replaced with the truncation + negative-correction form. Bilinear interpolation (`align_corners=False`, `pixel = loc*size - 0.5`) then matched CPU-canonical `F.grid_sample` across 9/9 T1 PASS. API-surface / compile claim, so the spec's `device.runtime: NOT_RUN` provenance flag does not block it (symbol absence is visible at compile, independent of device execution).
- **Other instances (predicted)**: any op with sub-pixel coordinate math — grid_sample, affine_grid, RoIAlign, deformable attention, NMS coordinate rounding, tiling `floor(N/tile)` when the quotient can be negative. Where the value is known non-negative (e.g. unsigned tiling counts), bare `static_cast<int32_t>` suffices; the correction is only needed where the operand can be < 0.
- **Cross-ref**: CAND-A5OPS-0004 (libm transcendentals unavailable at LINK — different stage + function class, fixed by SIMT/SIMD split, not a manual floor); [`EC-193`](ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md) (`__aicore__` forbids unsigned→float cast — same "device-TU restricts a plain C++ operation" family). backend=ascendc.

## 内嵌代码片段

### multi_scale_deformable_attn_function/multi_scale_deformable_attn_function_kernel.h.snippet.h

```cpp
                        continue;
                    }
                    int32_t wLow = FloorToInt(imW);
                    int32_t hLow = FloorToInt(imH);
                    float dW = imW - (float)wLow;
                    float dH = imH - (float)hLow;
                    float w1 = (1.0f - dH) * (1.0f - dW);
                    float w2 = (1.0f - dH) * dW;
```

**孪生卡 ec-265 合并的独有内容**（ec-265-floorf-floor-is-unavailable-in-the-aicore-kernel.md 已删除）：

> - **Error pattern**: at the ccec compile stage the kernel TU reports `floorf` (or `floor`) as undefined in the CCE kernel environment (`error: use of undeclared identifier 'floorf'`). This is the basic-math (`<cmath>`) floor symbol — NOT a transcendental. It is distinct from EC-35, which is a LINK-stage libm absence for transcendentals (`cosf`/`sinf`/`expf`/...) fixed by a SIMT→SIMD split; here the function is non-transcendental and the fix is a 2-line manual integer floor, not an extra kernel launch.

> int32_t x0 = static_cast<int32_t>(x_pix); // truncates toward zero

> if ((float)x0 > x_pix) x0--; // correct toward -inf for negatives

> - **Evidence**: multi_scale_deformable_attn_function 310p port (2026-08-10, Ascend310P3/dav-m200, arch 200x, CANN 9.0.0, port_a3_to_a5) — `target/ascendc/examples/multi_scale_deformable_attn_function/snippets.md` hit "floorf not available in AscendC kernel environment" at compile (logged `novel` in `multi_scale_deformable_attn_function/failures_ledger.md:4` iter 1); replaced with the truncation + negative-correction form. Bilinear interpolation (`align_corners=False`, `pixel = loc*size - 0.5`) then matched CPU-canonical `F.grid_sample` across 9/9 T1 PASS. API-surface / compile claim, so the spec's `device.runtime: NOT_RUN` provenance flag does not block it (symbol absence is visible at compile, independent of device execution).

> - **Cross-ref**: [`EC-35`](ec_35_aiv_device_binary_does_not_link_libm_transcendenta.md) (libm transcendentals unavailable at LINK — different stage + function class, fixed by SIMT/SIMD split, not a manual floor); [`EC-193`](ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md) (`__aicore__` forbids unsigned→float cast — same "device-TU restricts a plain C++ operation" family). backend=ascendc.
