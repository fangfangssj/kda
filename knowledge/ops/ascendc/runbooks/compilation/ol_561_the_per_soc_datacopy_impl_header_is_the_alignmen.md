---
type: Runbook
title: "The per-SoC DataCopy impl header IS the alignment/count contract — UB-src has a 32B assert, GM-dst has NONE, and unaligned counts silently round DOWN; never rely on the rounding, quantize counts yourself"
description: "per-SoC 的 DataCopy impl 头才是对齐/count 契约：UB 源有 32B 断言、GM 目的没有、非对齐 count 静默向下取整——count 要自己量子化。"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-561"]
---

# The per-SoC DataCopy impl header IS the alignment/count contract — UB-src has a 32B assert, GM-dst has NONE, and unaligned counts silently round DOWN; never rely on the rounding, quantize counts yourself

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any DataCopy UB↔GM path)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (22_HybridAttentionMaskPreparation kw-1, 2026-09-05 — SDK header read + 50/50 bit-exact device run)`
`unverified_on: soc=Ascend950PR, Ascend910_V220 (their impl headers assert differently — apply the same read-the-header step there, not these exact values)`

**Principle**: before designing around `DataCopy` alignment on a narrowed SoC, read the per-SoC impl header (CANN 安装树内 impl/basic_api/dav_m200/kernel_operator_data_copy_impl.h) — it encodes the real contract, and two of its three clauses are traps for 220x/351x intuition:
1. **UB-src address MUST be 32B-aligned** (asserted) — any UB staging buffer feeding a DataCopy must be allocated 32B-aligned.
2. **GM dst has NO alignment assert** — odd/unaligned GM row starts are legal. This is the header-level confirmation of the 13_Cat overlapping-tail idiom (`代码片段 13_Cat/kernel.h.snippet.h（见文末）`): you may open a write at an arbitrary GM offset.
3. **Unaligned counts are silently rounded DOWN** (a DataCopyCheck message only) — bytes past the rounding boundary are never written. Never rely on the rounding in either direction: quantize counts UP yourself and repair the tail (overlap-write the final aligned quantum), or make every count a whole 32B quantum by construction (PB-22 discipline).

**Concrete anchor**: per-run 32B-quantized uniform writes `DataCopy(gm+off, wd, ALIGN32(count))` with the ≤31B tail re-covered by the next write (13_Cat idiom); plane-end exactness from structural identities of the op (S=T+P ⇒ full-mask last row all-False, swa last row's True-run ends at S-1) — no data-dependent tail code.

- **Evidence**: 22_HybridAttentionMaskPreparation kw-1 (2026-09-05, Ascend310P3/dav-m200): SDK-confirmed contract used to design per-plane 32B-quantized mask writes including odd S=T+P row starts — 50/50 pass_a/pass_b bit-exact, det 50/50 3-run (`22_HybridAttentionMaskPreparation/verification.json`; device run, PB-74 precedent on the stale hardware-page flag).
- **Evidence (2nd, from the violating side)**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05, Ascend310P3/dav-m200): clause 3 confirmed on DEVICE by NOT following it — unaligned row-copy counts (fp32 S=100 → 400B) left the never-written stale tail and failed at max|dR|=1.75e+2 (fp16 400B → NaN) while every 32B-aligned S passed at 1e-8; fixed by aligned-prefix bulk copy + `__gm__` scalar tail. Full failure ladder and discrimination matrix in PB-127.
- **Other instances (predicted)**: every DataCopyPad-less (PB-105) ragged-tail port on arch-200; strided GM row writes with odd offsets (attention masks, padding/embedding tables, varlen packing); any `DataCopy(count)` where count % 8 (fp32) != 0.
- **Cross-ref**: [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (why plain DataCopy on dav-m200), [`PB-22`](pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md) (32B quantum), CAND-DAV-M200-UNALIGNED-DST-VEC (VEC-op dst alignment — a different surface, also legal unaligned), EC-123 (M-anchored overlapping tail store), [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (API lookup chain — the per-SoC impl-header check is its step-2.5 extension). backend=ascendc.

## 内嵌代码片段

### 13_Cat/kernel.h.snippet.h

```cpp
//
// Uses Adds(dst, src, 0) as a no-op VEC bridge (avoids PB-9 UB-to-UB
// DataCopy corruption). For non-aligned chunk tails, uses overlapping
// write technique to avoid overwriting adjacent data.

constexpr int32_t CAT_TILE_F32 = 4096;   // 4096 * 4B = 16KB
constexpr int32_t CAT_TILE_F16 = 4096;   // 4096 * 2B = 8KB (need room for f32 buf)
                    }
                }

                // Overlapping tail: read/write ALIGN elements starting at (chunk - ALIGN)
                // This re-copies some already-written elements but ensures no overwrite
                // past the chunk boundary.
                if (tail > 0 && chunk32 >= ALIGN) {
```

## 适用场景

- **工程形态**：per-SoC 的 DataCopy impl 头才是对齐/count 契约。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
