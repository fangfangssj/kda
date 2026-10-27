---
type: Runbook
title: "On dav-m200 WholeReduceSum's int32 mask is an element count, not a bitmask; srcRepStride is in 32B datablocks; the 7-arg 220x form doesn't compile on the 6-arg dav-m200 surface"
description: "On dav-m200 WholeReduceSum's int32 mask is an element count, not a bitmask; srcRepStride is in 32B datablocks; the 7-arg 220x form doesn't compile on the 6-arg dav-m200 surface"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["ReduceSum", "srcRepStride"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-416 — OL-416: A5Ops additive extension

> 关联：同一 OL-416 主题已由既有卡 [ol_416_on_dav_m200_wholereducesum_dst_src_mask_repeatti.md](../compilation/ol_416_on_dav_m200_wholereducesum_dst_src_mask_repeatti.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=reduction (row-sum / softmax / quant epilogue); dtype=fp32。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS; mask values 16/32/64 empirically pinned on device)。

**OL-416** — **on dav-m200 `WholeReduceSum(dst,src,mask,repeatTime,dstRepStride,srcBlkStride,srcRepStride)`'s int32 `mask` is an ELEMENT COUNT, not a bitmask (`SetMask<T>(len)`'s `(1<<len)-1` is the per-repeat vector-mask form); `srcRepStride` is in 32B datablocks; the 7-arg 220x form doesn't compile on the 6-arg dav-m200 surface.** Working row-sum `WholeReduceSum(rs, xF32, kSize_, TILE, 1, 1, kSize_/8)`, mask=kSize_ ∈ {16,32,64} (64 = FULL_MASK fast path). Anchor 内部路径（不公开）workspace/weight_quant_batch_matmul_v2/kernel/weight_quant_batch_matmul_v2_kernels.cpp:100. Evidence: weight_quant_batch_matmul_v2 310p 2026-08-11, mask semantics pinned empirically on device, 9/9 T1 PASS. Cross-ref CAND-DAV-M200-REDUCESUM-RETURNS-ZERO, [OL-407](../compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md)/346, EC-197, OL-81. backend=ascendc.

- Former A5Ops ID: `OL-416`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=reduction (row-sum / softmax / quant epilogue); dtype=fp32`
`verified_on: soc=Ascend310P3; cann=9.0.0 (weight_quant_batch_matmul_v2 310p, 2026-08-11 — 9/9 T1 PASS; mask values 16/32/64 empirically pinned on device)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the arch35/220x `WholeReduceSum` instances in this KB (online-softmax decompositions) do not state the mask convention; re-confirm per arch before transferring this element-count reading`

**Principle**: on dav-m200 the signature is `WholeReduceSum(dst, src, mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride)` and the int32 `mask` argument is the ELEMENT COUNT to reduce — not a bitmask and not a block count (empirically: mask=2→2 elements, 4→4, 8→8; `mask=kSize_` ∈ {16,32,64} works, and 64 hits the FULL_MASK fast path). This differs from `SetMask<T>(len)`, which produces a `(1<<len)-1` bitmask because the dav-m200 vector mask register covers 1 element per bit (for fp32). Also: `srcRepStride` is in 32-byte datablocks (vendor 归约计算API doc). A 7-arg 220x-form call (`&mask` by pointer) does not even match the dav-m200 6-arg surface — this is the EC-197 signature-drift class.

**Concrete anchor** (verified working row-sum, 内部路径（不公开）workspace/weight_quant_batch_matmul_v2/kernel/weight_quant_batch_matmul_v2_kernels.cpp:100):
```cpp
WholeReduceSum(rs, xF32, kSize_, TILE, 1, 1, kSize_ / 8);
//              |    |    |       |      |   |   └─ srcRepStride: kSize_ fp32 = kSize_/8 × 32B
//              |    |    |       |      |   └─ srcBlkStride
//              |    |    |       |      └─ dstRepStride
//              |    |    |       └─ repeatTime
//              |    |    └─ mask = ELEMENT COUNT (kSize_ ∈ {16,32,64}), NOT a bitmask
```

**Evidence**: weight_quant_batch_matmul_v2 310p (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5): iter-1 compile failure — 7-arg 220x-form `WholeReduceSum(rs, xF32, &mask, TILE, 1, 1, kSize_)` with `mask=kSize_/8` treated as a bitmask vs the dav-m200 6-arg signature (内部路径（不公开）workspace/weight_quant_batch_matmul_v2/failures_ledger.md:4, logged `novel`); mask semantics pinned empirically on device (mask=2→2 elems, 4→4, 8→8). After fix (`mask=kSize_`, `srcRepStride=kSize_/8`): 9/9 T1 PASS, det 9/9 bit-exact.

**Other instances (predicted)**: any dav-m200 per-row reduction — softmax row-max/row-sum, quant per-row scale/offset sums, norm row statistics, dot-product rows in grad kernels. When porting a 220x/A3-source reduce to dav-m200, re-derive BOTH the argument count and the mask unit from the 9.0.0 header before copying the call.

**Cross-ref**: CAND-DAV-M200-REDUCESUM-RETURNS-ZERO (the `ReduceSum`-returns-zero sibling — the same reduce-primitive flavor chain on dav-m200: pick primitives by empirically-pinned semantics, not by surface similarity), OL-407 + OL-415 (the "dav-m200 primitive args are element counts / datablocks, not bytes/bitmasks" family), EC-197 (signature drift between CANN versions), OL-81 (the CAST chain — the precision-side sibling of this op's epilogue). backend=ascendc.
