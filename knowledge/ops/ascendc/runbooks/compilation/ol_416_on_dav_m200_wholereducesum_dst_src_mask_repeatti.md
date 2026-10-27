---
type: Runbook
title: "on dav-m200 WholeReduceSum(dst,src,mask,repeatTime,dstRepStride,srcBlkStride,srcRepStride)'s int32 mask is an"
description: "dav-m200 WholeReduceSum 的 mask 参数是元素个数而非位掩码，srcRepStride 单位是 32B 块；220x 的 7 参形态在此不兼容"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.compute.reduction, topic.implementation.masking, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-416"]
---

# on dav-m200 WholeReduceSum(dst,src,mask,repeatTime,dstRepStride,srcBlkStride,srcRepStride)'s int32 mask is an

**OL-416** — **on dav-m200 `WholeReduceSum(dst,src,mask,repeatTime,dstRepStride,srcBlkStride,srcRepStride)`'s int32 `mask` is an ELEMENT COUNT, not a bitmask (`SetMask<T>(len)`'s `(1<<len)-1` is the per-repeat vector-mask form); `srcRepStride` is in 32B datablocks; the 7-arg 220x form doesn't compile on the 6-arg dav-m200 surface.** Working row-sum `WholeReduceSum(rs, xF32, kSize_, TILE, 1, 1, kSize_/8)`, mask=kSize_ ∈ {16,32,64} (64 = FULL_MASK fast path). Anchor `代码片段 weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]`. Evidence: weight_quant_batch_matmul_v2 310p 2026-08-11, mask semantics pinned empirically on device, 9/9 T1 PASS. Cross-ref, [`OL-407`](ol_407_datacopy_count_is_an_element_count_on_every_arch.md)/346, [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md), [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=reduction (row-sum / softmax / quant epilogue); dtype=fp32`
`verified_on: soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS; mask values 16/32/64 empirically pinned on device)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the arch35/220x `WholeReduceSum` instances in this KB (online-softmax decompositions) do not state the mask convention; re-confirm per arch before transferring this element-count reading`

**Principle**: on dav-m200 the signature is `WholeReduceSum(dst, src, mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride)` and the int32 `mask` argument is the ELEMENT COUNT to reduce — not a bitmask and not a block count (empirically: mask=2→2 elements, 4→4, 8→8; `mask=kSize_` ∈ {16,32,64} works, and 64 hits the FULL_MASK fast path). This differs from `SetMask<T>(len)`, which produces a `(1<<len)-1` bitmask because the dav-m200 vector mask register covers 1 element per bit (for fp32). Also: `srcRepStride` is in 32-byte datablocks (vendor 归约计算API doc). A 7-arg 220x-form call (`&mask` by pointer) does not even match the dav-m200 6-arg surface — this is the EC-197 signature-drift class.

**Concrete anchor** (verified working row-sum, `代码片段 weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]`):
```cpp
WholeReduceSum(rs, xF32, kSize_, TILE, 1, 1, kSize_ / 8);
//              |    |    |       |      |   |   └─ srcRepStride: kSize_ fp32 = kSize_/8 × 32B
//              |    |    |       |      |   └─ srcBlkStride
//              |    |    |       |      └─ dstRepStride
//              |    |    |       └─ repeatTime
//              |    |    └─ mask = ELEMENT COUNT (kSize_ ∈ {16,32,64}), NOT a bitmask
```

**Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): iter-1 compile failure — 7-arg 220x-form `WholeReduceSum(rs, xF32, &mask, TILE, 1, 1, kSize_)` with `mask=kSize_/8` treated as a bitmask vs the dav-m200 6-arg signature (`weight_quant_batch_matmul_v2/failures_ledger.md:4`, logged `novel`); mask semantics pinned empirically on device (mask=2→2 elems, 4→4, 8→8). After fix (`mask=kSize_`, `srcRepStride=kSize_/8`): 9/9 T1 PASS, det 9/9 bit-exact.

**Other instances (predicted)**: any dav-m200 per-row reduction — softmax row-max/row-sum, quant per-row scale/offset sums, norm row statistics, dot-product rows in grad kernels. When porting a 220x/A3-source reduce to dav-m200, re-derive BOTH the argument count and the mask unit from the 9.0.0 header before copying the call.

**Cross-ref**: (the `ReduceSum`-returns-zero sibling — the same reduce-primitive flavor chain on dav-m200: pick primitives by empirically-pinned semantics, not by surface similarity), [`OL-407`](ol_407_datacopy_count_is_an_element_count_on_every_arch.md) + [`OL-415`](ol_415_datacopyparams_stride_slots_are_direction_asymme.md) (the "dav-m200 primitive args are element counts / datablocks, not bytes/bitmasks" family), [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) (signature drift between CANN versions), [`OL-81`](../precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md) (the CAST chain — the precision-side sibling of this op's epilogue). backend=ascendc.

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

**孪生卡 ol-643 合并的独有内容**（ol-643-on-dav-m200-wholereducesum-s-int32-mask-paramete.md 已删除）：

> **Concrete anchor** (verified working row-sum, `target/ascendc/examples/weight_quant_batch_matmul_v2/snippets.md`):

> // | | └─ mask = ELEMENT COUNT (kSize_ ∈ {16,32,64}), NOT a bitmask

> **Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): iter-1 compile failure — 7-arg 220x-form `WholeReduceSum(rs, xF32, &mask, TILE, 1, 1, kSize_)` with `mask=kSize_/8` treated as a bitmask vs the dav-m200 6-arg signature (`weight_quant_batch_matmul_v2/failures_ledger.md:4`, logged `novel`); mask semantics pinned empirically on device (mask=2→2 elems, 4→4, 8→8). After fix (`mask=kSize_`, `srcRepStride=kSize_/8`): 9/9 T1 PASS, det 9/9 bit-exact.
