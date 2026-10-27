---
type: Runbook
title: "DataCopyParams stride slots are direction-asymmetric and in 32-byte datablocks on dav-m200 (never byte units"
description: "dav-m200 的 DataCopyParams stride 槽位方向不对称且单位是 32B 块（不是字节）：读走 srcStride、写走 dstStride 槽"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.cube, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-415"]
---

# DataCopyParams stride slots are direction-asymmetric and in 32-byte datablocks on dav-m200 (never byte units

**OL-415** — **`DataCopyParams` stride slots are direction-asymmetric and in 32-byte datablocks on dav-m200 (never byte units, never symmetric src/dst roles): reads advance via `srcStride` in 32B datablocks; writes advance via the DSTSTRIDE slot, also in 32B datablocks (`dstStride=32` ⇒ 1024B/burst).** Plus sub-principle: never 32B-DataCopy a sub-row int8 window at an odd 16-boundary — load the full row, window on the fp16 side. Anchors `代码片段 weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]` (`DataCopyParams(TILE,1,0,(nSize_-TILE)/TILE)` store form). Evidence: weight_quant_batch_matmul_v2 310p 2026-08-11, 9/9 T1 PASS + det 9/9 (empirical NPU stride sweep). Cross-ref [`OL-407`](ol_407_datacopy_count_is_an_element_count_on_every_arch.md), [`OL-279`](ol_279_fixpipe_l0c_gm_dststride_unit_is_per_format_elemen.md), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=data-movement / cube-epilogue (strided row copies); dtype=fp16/fp32/int8`
`verified_on: soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS after the stride-unit + slot fixes)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — 220x/351x `DataCopyParams` stride semantics are NOT known to differ; the measured asymmetry/units are dav-m200-anchored. See OL-279 for the 220x `Fixpipe` parallel (stride units are per-API+format there too — same "never assume units" family).`

**Principle**: `DataCopyParams(blockCount, blockLen, srcStride, dstStride)` on dav-m200 has NEITHER symmetric src/dst stride roles NOR byte units:
- **GM→UB/L1 reads**: `srcStride` is the gap between bursts in 32-byte datablocks (`srcGap = srcStride × 32`, per the `copy_gm_to_ubuf` semantics).
- **UB→GM writes**: the DSTSTRIDE slot is what advances the destination across bursts. Putting the store stride in the `srcStride` slot reads source garbage — the earlier bug landed rows of garbage because the store stride never advanced the destination.
- **dstStride unit = 32-byte datablocks, NOT bytes**: an empirical NPU sweep showed `dstStride=32` advancing 1024 B per burst (only row 0 of the tile landed in range). The wrapper's `dstGap=dstStride` reading is a mislead — the aligned-variant downstream applies the ×32.

**Sub-principle (int8 sub-row windows)**: a 16-column int8 chunk at column n0=16 starts at BYTE offset 16 inside a 32-byte row → 32B-misaligned for DataCopy. Fix pattern: load the FULL row (aligned), dequantize to fp16, select the 16-col window on the fp16 side (16 halfs = 32B ⇒ aligned). Kernel-generic: never 32B-DataCopy a sub-row int8 window at an odd 16-boundary.

**Concrete anchor** (verified working forms, `代码片段 weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]`):
```cpp
// strided GM→L1 read: TILE bursts × 32B, inter-burst gap (kSize_/TILE - 1) in 32B datablocks
DataCopy(xL1, xGm_[m0*kSize_ + kc*TILE], DataCopyParams(TILE, 1, (kSize_/TILE) - 1, 0));   // :176
// strided UB→GM store of fp16 rows: dstStride goes in SLOT 4, in 32B datablocks
DataCopy(yGm_[m0*nSize_ + n0], y16, DataCopyParams(TILE, 1, 0, (nSize_-TILE)/TILE));      // :301
```

**Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): (a) iter-5 precision failure (rows 1–15 wrong) — store stride placed in the srcStride slot AND in the wrong unit; fixed to `(16, 1, 0, (nSize_-16)/16)` 32B units (`weight_quant_batch_matmul_v2/failures_ledger.md:11`, logged `novel`); (b) iter-3 failure — int8 16-col chunk at n0=16 misaligned → full-row load + fp16-side windowing (`weight_quant_batch_matmul_v2/failures_ledger.md:9`); (c) iter-1 compile failure — x-chunk `DataCopyParams(1,16,stride,0)` contiguous-burst form wrong for K>16, fixed to the strided `(16,1,(K/16)-1,0)` (`weight_quant_batch_matmul_v2/failures_ledger.md:3`). After fixes: 9/9 T1 PASS, det 9/9 bit-exact.

**Other instances (predicted)**: any dav-m200 port with strided row copies — cube operand staging (this op), conv-im2col staging, transpose-by-stride epilogues, padded-row stores. Grep the port source for `DataCopyParams` and re-derive every stride slot+unit from the dav-m200 headers before trusting a 220x/A3-source form.

**Cross-ref**: [`OL-407`](ol_407_datacopy_count_is_an_element_count_on_every_arch.md) (the "primitive args are element counts, not bytes" sibling — same family of dav-m200 unit traps), [`OL-279`](ol_279_fixpipe_l0c_gm_dststride_unit_is_per_format_elemen.md) (220x `Fixpipe` stride-unit-per-format — the cross-arch "units are per-API" parallel), [`PB-105`](pb_105_datacopypad_datacopyextparams_are_not_supported.md) (dav-m200 DataCopyPad/ExtParams unsupported — the plain-DataCopy fallback this entry's params belong to), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) (CANN 9.0.0 signature drift — verify every form against 9.0.0 headers). backend=ascendc.

## 内嵌代码片段

### weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_kernels.cpp.snippet.cpp

```cpp
    // doc; probe-confirmed): row r at xF32[r*kSize_] = r*(kSize_/8) 32B-blocks.
    // dst advance unit = 1 reduced value → dstRepStride=1 ⇒ rs[r].
    LocalTensor<float> rs = rsBuf_.template Get<float>();
    WholeReduceSum(rs, xF32, kSize_, TILE, 1, 1, kSize_ / 8);
    PipeBarrier<PIPE_ALL>();  // V→S for later scalar GetValue
}

        // inter-burst gap (kSize_/16 - 1)×32B between rows (probe4 form for
        // contiguous K=16; strided form for K=32/64).
        uint32_t xSrcStride = (kSize_ / TILE) - 1;  // 32B units between 32B bursts
        DataCopy(xL1, xGm_[m0 * kSize_ + kc * TILE], DataCopyParams(TILE, 1, xSrcStride, 0));
        // w chunk [16×16] fp16 window at column n0 of the per-tile staged full
        // row [16×nSize_] (fp16 ⇒ 16 halfs = 32B ⇒ n0=16 stays aligned).
        DataCopy(wL1, wsGm_[(tileIdx_ * kChunks_ + kc) * TILE * nSize_ + n0],
        // store stride goes in the DSTSTRIDE slot (bytes for UB→GM); the earlier
        // (TILE,1,dstStride,0) put it in srcStride → src advanced 32B×stride per
        // burst → garbage (masked by 0-stride on N=16, exposed on N=32).
        DataCopy(yGm_[m0 * nSize_ + n0], y16, DataCopyParams(TILE, 1, 0, dstStride));
    } else {
        // int8 output (ungraded): y = clamp(round(acc*qs + qo), -128, 127)
        LocalTensor<float> qs32 = sp32[5 * TILE];
```

**孪生卡 ol-642 合并的独有内容**（ol-642-datacopyparams-stride-slots-are-direction-asymme.md 已删除）：

> **Concrete anchor** (verified working forms, `target/ascendc/examples/weight_quant_batch_matmul_v2/snippets.md`):

> // strided GM→L1 read: TILE bursts × 32B, inter-burst gap (kSize_/TILE - 1) in 32B datablocks

> DataCopy(xL1, xGm_[m0*kSize_ + kc*TILE], DataCopyParams(TILE, 1, (kSize_/TILE) - 1, 0)); // :176

> // strided UB→GM store of fp16 rows: dstStride goes in SLOT 4, in 32B datablocks

> DataCopy(yGm_[m0*nSize_ + n0], y16, DataCopyParams(TILE, 1, 0, (nSize_-TILE)/TILE)); // :301

> **Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): (a) iter-5 precision failure (rows 1–15 wrong) — store stride placed in the srcStride slot AND in the wrong unit; fixed to `(16, 1, 0, (nSize_-16)/16)` 32B units (`weight_quant_batch_matmul_v2/failures_ledger.md:11`, logged `novel`); (b) iter-3 failure — int8 16-col chunk at n0=16 misaligned → full-row load + fp16-side windowing (`weight_quant_batch_matmul_v2/failures_ledger.md:9`); (c) iter-1 compile failure — x-chunk `DataCopyParams(1,16,stride,0)` contiguous-burst form wrong for K>16, fixed to the strided `(16,1,(K/16)-1,0)` (`weight_quant_batch_matmul_v2/failures_ledger.md:3`). After fixes: 9/9 T1 PASS, det 9/9 bit-exact.

## 补充证据（2026-10-10 合并自 `datacopyparams_strides_direction_asymmetric_32b_datablocks.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-415 — OL-415: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=data-movement / cube-epilogue (strided row copies); dtype=fp16/fp32/int8。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS after the stride-unit + slot fixes)。

**OL-415** — **`DataCopyParams` stride slots are direction-asymmetric and in 32-byte datablocks on dav-m200 (never byte units, never symmetric src/dst roles): reads advance via `srcStride` in 32B datablocks; writes advance via the DSTSTRIDE slot, also in 32B datablocks (`dstStride=32` ⇒ 1024B/burst).** Plus sub-principle: never 32B-DataCopy a sub-row int8 window at an odd 16-boundary — load the full row, window on the fp16 side. Anchors 内部路径（不公开）workspace/weight_quant_batch_matmul_v2/kernel/weight_quant_batch_matmul_v2_kernels.cpp:176,301 (`DataCopyParams(TILE,1,0,(nSize_-TILE)/TILE)` store form). Evidence: weight_quant_batch_matmul_v2 310p 2026-08-11, 9/9 T1 PASS + det 9/9 (empirical NPU stride sweep). Cross-ref [OL-407](ol_407_datacopy_count_is_an_element_count_on_every_arch.md), [OL-279](ol_279_fixpipe_l0c_gm_dststride_unit_is_per_format_elemen.md), [PB-105](dav_m200_datacopypad_unsupported_both_directions.md), [EC-197](cann_9_0_0_dav_m200_mask_form_vec_signatures.md). backend=ascendc.

- Former A5Ops ID: `OL-415`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=data-movement / cube-epilogue (strided row copies); dtype=fp16/fp32/int8`
`verified_on: soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS after the stride-unit + slot fixes)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — 220x/351x `DataCopyParams` stride semantics are NOT known to differ; the measured asymmetry/units are dav-m200-anchored. See OL-279 for the 220x `Fixpipe` parallel (stride units are per-API+format there too — same "never assume units" family).`

**Principle**: `DataCopyParams(blockCount, blockLen, srcStride, dstStride)` on dav-m200 has NEITHER symmetric src/dst stride roles NOR byte units:
- **GM→UB/L1 reads**: `srcStride` is the gap between bursts in 32-byte datablocks (`srcGap = srcStride × 32`, per the `copy_gm_to_ubuf` semantics).
- **UB→GM writes**: the DSTSTRIDE slot is what advances the destination across bursts. Putting the store stride in the `srcStride` slot reads source garbage — the earlier bug landed rows of garbage because the store stride never advanced the destination.
- **dstStride unit = 32-byte datablocks, NOT bytes**: an empirical NPU sweep showed `dstStride=32` advancing 1024 B per burst (only row 0 of the tile landed in range). The wrapper's `dstGap=dstStride` reading is a mislead — the aligned-variant downstream applies the ×32.

**Sub-principle (int8 sub-row windows)**: a 16-column int8 chunk at column n0=16 starts at BYTE offset 16 inside a 32-byte row → 32B-misaligned for DataCopy. Fix pattern: load the FULL row (aligned), dequantize to fp16, select the 16-col window on the fp16 side (16 halfs = 32B ⇒ aligned). Kernel-generic: never 32B-DataCopy a sub-row int8 window at an odd 16-boundary.

**Concrete anchor** (verified working forms, `workspace/weight_quant_batch_matmul_v2/kernel/weight_quant_batch_matmul_v2_kernels.cpp`):
```cpp
// strided GM→L1 read: TILE bursts × 32B, inter-burst gap (kSize_/TILE - 1) in 32B datablocks
DataCopy(xL1, xGm_[m0*kSize_ + kc*TILE], DataCopyParams(TILE, 1, (kSize_/TILE) - 1, 0));   // :176
// strided UB→GM store of fp16 rows: dstStride goes in SLOT 4, in 32B datablocks
DataCopy(yGm_[m0*nSize_ + n0], y16, DataCopyParams(TILE, 1, 0, (nSize_-TILE)/TILE));      // :301
```

**Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): (a) iter-5 precision failure (rows 1–15 wrong) — store stride placed in the srcStride slot AND in the wrong unit; fixed to `(16, 1, 0, (nSize_-16)/16)` 32B units (内部路径（不公开）workspace/weight_quant_batch_matmul_v2/failures_ledger.md:11, logged `novel`); (b) iter-3 failure — int8 16-col chunk at n0=16 misaligned → full-row load + fp16-side windowing (内部路径（不公开）workspace/weight_quant_batch_matmul_v2/failures_ledger.md:9); (c) iter-1 compile failure — x-chunk `DataCopyParams(1,16,stride,0)` contiguous-burst form wrong for K>16, fixed to the strided `(16,1,(K/16)-1,0)` (内部路径（不公开）workspace/weight_quant_batch_matmul_v2/failures_ledger.md:3). After fixes: 9/9 T1 PASS, det 9/9 bit-exact.

**Other instances (predicted)**: any dav-m200 port with strided row copies — cube operand staging (this op), conv-im2col staging, transpose-by-stride epilogues, padded-row stores. Grep the port source for `DataCopyParams` and re-derive every stride slot+unit from the dav-m200 headers before trusting a 220x/A3-source form.

**Cross-ref**: OL-407 (the "primitive args are element counts, not bytes" sibling — same family of dav-m200 unit traps), OL-279 (220x `Fixpipe` stride-unit-per-format — the cross-arch "units are per-API" parallel), PB-105 (dav-m200 DataCopyPad/ExtParams unsupported — the plain-DataCopy fallback this entry's params belong to), EC-197 (CANN 9.0.0 signature drift — verify every form against 9.0.0 headers). backend=ascendc.
