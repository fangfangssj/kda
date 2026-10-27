---
type: Runbook
title: "hardware gather (Gather/vgather) is the right primitive for ARBITRARY-index gather on SIMD-only arches (dav-m2"
description: "任意索引 gather 在纯 SIMD 架构上的正确原语是硬件 Gather/vgather（310P3 的 ini 列出六种 dtype 支持）——注意偏移是字节单位、Gatherb 是废弃 stub"
tags: [task.optimization, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation, topic.implementation.masking, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-552"]
---

# hardware gather (Gather/vgather) is the right primitive for ARBITRARY-index gather on SIMD-only arches (dav-m2

## 优化点

**OL-552** — hardware gather (`Gather`/vgather) is the right primitive for ARBITRARY-index gather on SIMD-only arches (dav-m200 verified: Ascend310P3.ini lists `Intrinsic_vgather` for int16/uint16/fp16/fp32/int32/uint32). Three semantic gotchas: offsets are BYTE offsets relative to `src base + srcBaseAddr` (pre-scale element indices by `sizeof(T)`), the Level-2 count-form API auto-decomposes into repeat batches (no manual repeat bookkeeping), `Gatherb` is a deprecated stub. Discriminator vs strided movement: no expressible stride structure → Gather; strided/tiled → `DataCopyParams`/`Transpose` (CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN). int16 coverage lets bf16 ride the same template (CAND-DAV-M200-INT16-BF16-DATAMOVE). Gather is a long-V-latency consumer — its row loop needs the PB-49 V→MTE2 fence. 18_Index 310p 2026-08-15: 41/41 + 201/201 bit-exact, det 3×41, median 0.79×. backend=ascendc.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=gather-class data movement (index_select / gather / embedding-index read) with arbitrary, non-strided indices`
`verified_on: soc=Ascend310P3; cann=9.0.0 (18_Index 310p, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — vgather AVAILABILITY is per-arch (check each SoC's <SoC>.ini Intrinsic_vgather dtype list before relying on it); the byte-offset / count-API semantics are expected to carry but were not exercised there`

### Principle

For gather-class data movement whose index set is ARBITRARY (index_select / gather — no strided or tiled structure a `DataCopyParams` or `Transpose` can express), the hardware `Gather` VEC primitive is the right primitive on arches that list it; a per-element scalar `GetValue` loop is the documented catastrophic anti-pattern. Three semantic gotchas, verified on dav-m200:

1. **Offsets are BYTE offsets**, relative to `src base + srcBaseAddr` — an element-indexed offset table must be pre-scaled by `sizeof(T)` before the call; forgetting the scale gathers silently from the wrong elements with no error.
2. The **Level-2 count-form API auto-decomposes** into repeat batches — no manual repeat-stride bookkeeping for counts beyond one repeat's worth.
3. **`Gatherb` (block gather) is a deprecated stub** on this arch — do not use; the element-gather spelling is the live one.

Availability is declared per-arch in the SoC `.ini` (`Intrinsic_vgather`): on Ascend310P3 it covers int16/uint16/fp16/fp32/int32/uint32 — the int16 coverage is what lets a bf16-carrying data-movement kernel reuse the identical template path with zero casts (CAND-DAV-M200-INT16-BF16-DATAMOVE).

**Discriminator vs strided movement**: if the index set has expressible stride/tiling structure (permute/transpose/slice/repeat), take the strided `DataCopyParams` / tiled `Transpose` route instead (CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN); hardware gather is for the arbitrary-index residue.

### Concrete anchor

```cpp
// byte-offset table (built scalar-side or host-side), one call per destination tile
Gather(outT, rowD, off[static_cast<uint32_t>(pos)], static_cast<uint32_t>(0),
       static_cast<uint32_t>(count));  // Level-2 count form — auto repeat-decomposed
```
Anchor: `18_Index/kernel/index_kernel.h:185 [snippet-unavailable: source workspace not on this box]` (call site) + `18_Index/kernel/index_tiling.h:12 [snippet-unavailable: source workspace not on this box]` (mode-selection comment); API semantics per vendor doc CANN 安装树内 atlasascendc_api_07_0091 + the CANN 安装树内 Ascend310P3.ini Intrinsic_vgather dtype list.

### Evidence

- 18_Index 310p kw-1→kw-2 (2026-08-15, benchmark, `torch.index_select`): MODE_A row-cache + hardware Gather — 41/41 pass A + 201/201 pass B bit-exact (fp32 16/16, fp16 17/17, bf16 8/8 via the int16 carrier), det 3×41 identical; perf mean 3.58× / median 0.79× vs torch (one 76.8× outlier disclosed as vendor-ref AICPU-fallback skew, with mean-without-skew 1.75× reported alongside — not headlined). First in-project use of hardware vgather on dav-m200. **Provenance**: device UNVERIFIED — [target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` on this branch.
- Discovery chain: the primitive was located via the OL-80/OL-130 API-lookup chain (SDK header + vendor doc page 0091) before use.

### Cross-reference

- CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN (the perf anti-pattern this primitive avoids; also the strided-vs-arbitrary discriminator).
- CAND-DAV-M200-INT16-BF16-DATAMOVE (int16 dtype coverage → bf16 rides the same template).
- PB-49 (the V→MTE2 fence a Gather consumer's row loop needs — Gather's long V latency widens that race window).
- OL-80 / OL-130 (the API lookup chain used to find and verify the primitive).

## 补充证据（2026-10-10 合并自 `hardware_gather_arbitrary_index_simd_only_arches.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-552 — OL-552: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=gather-class data movement (index_select / gather / embedding-index read) with arbitrary, non-strided indices。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (18_Index 310p, 2026-08-15)。

**OL-552** — hardware gather (`Gather`/vgather) is the right primitive for ARBITRARY-index gather on SIMD-only arches (dav-m200 verified: Ascend310P3.ini lists `Intrinsic_vgather` for int16/uint16/fp16/fp32/int32/uint32). Three semantic gotchas: offsets are BYTE offsets relative to `src base + srcBaseAddr` (pre-scale element indices by `sizeof(T)`), the Level-2 count-form API auto-decomposes into repeat batches (no manual repeat bookkeeping), `Gatherb` is a deprecated stub. Discriminator vs strided movement: no expressible stride structure → Gather; strided/tiled → `DataCopyParams`/`Transpose` (CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN). int16 coverage lets bf16 ride the same template (CAND-DAV-M200-INT16-BF16-DATAMOVE). Gather is a long-V-latency consumer — its row loop needs the PB-49 V→MTE2 fence. 18_Index 310p 2026-08-15: 41/41 + 201/201 bit-exact, det 3×41, median 0.79×. backend=ascendc.

- Former A5Ops ID: `OL-552`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=gather-class data movement (index_select / gather / embedding-index read) with arbitrary, non-strided indices`
`verified_on: soc=Ascend310P3; cann=9.0.0 (18_Index 310p, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — vgather AVAILABILITY is per-arch (check each SoC's <SoC>.ini Intrinsic_vgather dtype list before relying on it); the byte-offset / count-API semantics are expected to carry but were not exercised there`

### Principle

For gather-class data movement whose index set is ARBITRARY (index_select / gather — no strided or tiled structure a `DataCopyParams` or `Transpose` can express), the hardware `Gather` VEC primitive is the right primitive on arches that list it; a per-element scalar `GetValue` loop is the documented catastrophic anti-pattern. Three semantic gotchas, verified on dav-m200:

1. **Offsets are BYTE offsets**, relative to `src base + srcBaseAddr` — an element-indexed offset table must be pre-scaled by `sizeof(T)` before the call; forgetting the scale gathers silently from the wrong elements with no error.
2. The **Level-2 count-form API auto-decomposes** into repeat batches — no manual repeat-stride bookkeeping for counts beyond one repeat's worth.
3. **`Gatherb` (block gather) is a deprecated stub** on this arch — do not use; the element-gather spelling is the live one.

Availability is declared per-arch in the SoC `.ini` (`Intrinsic_vgather`): on Ascend310P3 it covers int16/uint16/fp16/fp32/int32/uint32 — the int16 coverage is what lets a bf16-carrying data-movement kernel reuse the identical template path with zero casts (CAND-DAV-M200-INT16-BF16-DATAMOVE).

**Discriminator vs strided movement**: if the index set has expressible stride/tiling structure (permute/transpose/slice/repeat), take the strided `DataCopyParams` / tiled `Transpose` route instead (CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN); hardware gather is for the arbitrary-index residue.

### Concrete anchor

```cpp
// byte-offset table (built scalar-side or host-side), one call per destination tile
Gather(outT, rowD, off[static_cast<uint32_t>(pos)], static_cast<uint32_t>(0),
       static_cast<uint32_t>(count));  // Level-2 count form — auto repeat-decomposed
```
Anchor: 内部路径（不公开）workspace/18_Index/kernel/index_kernel.h:185 (call site) + 内部路径（不公开）workspace/18_Index/kernel/index_tiling.h:12 (mode-selection comment); API semantics per vendor doc CANN 安装树内 atlasascendc_api_07_0091 + the CANN 安装树内 Ascend310P3.ini Intrinsic_vgather dtype list.

### Evidence

- 18_Index 310p kw-1→kw-2 (2026-08-15, benchmark, `torch.index_select`): MODE_A row-cache + hardware Gather — 41/41 pass A + 201/201 pass B bit-exact (fp32 16/16, fp16 17/17, bf16 8/8 via the int16 carrier), det 3×41 identical; perf mean 3.58× / median 0.79× vs torch (one 76.8× outlier disclosed as vendor-ref AICPU-fallback skew, with mean-without-skew 1.75× reported alongside — not headlined). First in-project use of hardware vgather on dav-m200. **Provenance**: device UNVERIFIED — `hardware/target/ascend310p.md` declares `device.runtime: NOT_RUN` / `compile_evidence_device_execution: FORBIDDEN` on this branch.
- Discovery chain: the primitive was located via the OL-80/OL-130 API-lookup chain (SDK header + vendor doc page 0091) before use.

### Other instances (predicted)

- Any arbitrary-index read op on SIMD-only 200x-class arches: `torch.gather` variants, embedding-lookup index reads, row assembly for one-hot / scatter-read kernels.
- NOT applicable to strided/tiled movement (permute/transpose/slice/repeat) — that class keeps the strided `DataCopyParams` / `Transpose` route; the discriminator is whether the index set has expressible stride structure.

### Cross-reference

- CAND-SCALAR-GATHER-BULK-STRIDED-ANTIPATTERN (the perf anti-pattern this primitive avoids; also the strided-vs-arbitrary discriminator).
- CAND-DAV-M200-INT16-BF16-DATAMOVE (int16 dtype coverage → bf16 rides the same template).
- PB-49 (the V→MTE2 fence a Gather consumer's row loop needs — Gather's long V latency widens that race window).
- OL-80 / OL-130 (the API lookup chain used to find and verify the primitive).

---
