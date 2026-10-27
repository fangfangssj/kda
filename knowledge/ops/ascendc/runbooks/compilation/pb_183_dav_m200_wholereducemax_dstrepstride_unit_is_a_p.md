---
type: Runbook
title: "dav-m200 `WholeReduceMax` `dstRepStride` unit is a PAIR (value,idx,pad,pad) while `WholeReduceSum`'s is an ELEMENT — the two Level-2 WholeReduce APIs do NOT share a stride-unit convention and neither layout is documented in the installed headers [dav-m200/Ascend310P3]"
description: "m200 `vcmax` 的 ORDER_VALUE_INDEX 模式按 (value,index,pad,pad) 写，且 `dstRepStride` 以对为单位——Level-2 封装的 stride 语义与直觉不同。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.compute.reduction, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-183", "WholeReduceMax", "dstRepStride", "WholeReduceSum"]
---

# dav-m200 `WholeReduceMax` `dstRepStride` unit is a PAIR (value,idx,pad,pad) while `WholeReduceSum`'s is an ELEMENT — the two Level-2 WholeReduce APIs do NOT share a stride-unit convention and neither layout is documented in the installed headers [dav-m200/Ascend310P3]

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any reduce-then-broadcast chain: softmax / attention / gating that reduces an axis and re-broadcasts row stats)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (27_MultiMaskAttentionAggregation kw-1 layout probe iter 4, 2026-09-04)`
`unverified_on: soc=Ascend950PR (351x) — different API generation there (ReduceOrder exists; see OL-195); soc=Ascend910 (220x)`

- **Severity**: HIGH — silent garbage broadcast seed; no trap, compiles clean.
- **Status**: CONFIRMED 2026-09-04 (single-fold layout micro-test + full-shape exactness verification).
- **Symptom**: m200 `vcmax` in ORDER_VALUE_INDEX mode writes (value, index, pad, pad) per repeat, and the `dstRepStride` field of the Level-2 `WholeReduceMax(dst, src, mask, repeatTime, dstRepStride, srcBlkStride, srcRepStride)` is expressed in PAIRS: `dstRepStride=1` packs pairs (values land at `dst[2v]`); `dstRepStride=2` spaces them at `dst[4v]`, leaving idx/pad slots at the odd multiples that a value-slot Gather (byte offset `8*v`) then reads as the row max → `exp(x − garbage)` → inf/NaN or silently wrong softmax.
- **Mechanism**: the dst-layout convention differs BETWEEN the two WholeReduce APIs on the same SoC: `WholeReduceSum`'s `dstRepStride` unit is an ELEMENT (`dstRepStride=2` puts values at `dst[2v]` — 7_Sum-proven, re-verified here). Copying the Sum idiom onto the Max call (the exact observed failure: kw-1 spawn #1 iters 1-3, fp32 row-maxes garbage) therefore lands values at the wrong stride. There is no order-selection escape hatch on this surface: `ReduceOrder::ORDER_VALUE` is ABSENT on m200 (compile error), so the pair layout MUST be handled, not avoided — contrast the 351x fix axis in OL-195.
- **Fix**: keep `dstRepStride=1` (PAIR unit) so values sit at `dst[2v]`, and derive the broadcast Gather's byte-offset table from the empirically pinned slot layout (`8*v` bytes for value slot v). Anchor: `代码片段 27_MultiMaskAttentionAggregation/mmagg_kernel.h.snippet.h（见文末）`. Probe the slot layout once per new (op, dtype) — it is not documented in the installed headers' doc comments.
- **Evidence**: 27_MultiMaskAttentionAggregation 310p (2026-09-04): layout probe iter 4 pinned (value,idx,pad,pad) per repeat; with `dstRepStride=1` row-maxes verified exact vs CPU rowmax on every probed shape N=64..512; rowsum side z-exact (7_Sum ReduceChainA reused verbatim — `reduce_sum_kernel.h`（a5_ops 路径，未迁移）). Full op: Pass A 50/50 T1, Pass B 93/93 T1, det 50/50 bit-exact.
- **Other instances (predicted)**: any m200 kernel that reduces row stats with Max then broadcasts (online-softmax, per-row gating, per-row amax for quant) — especially ports that copy a Sum-proven stride idiom onto the Max call.
- **Cross-ref**: [`OL-195`](ol_195_v220_v351_vec_compute_chain_port_pipebarrier_pipe.md) (351x sibling of the same packing family — fix axis there is `ReduceOrder::ORDER_ONLY_VALUE`, unavailable here), OL-347 (dav-m200 `WholeReduceSum` mask=count semantics), P-P129 (byte-offset Gather broadcast), [`PB-21`](pb_21_pipebarrier_pipe_all_between_tbuf_resident_mte2_lo.md) (same op's arch-220x-era 507015 — different failure). backend=ascendc.

## 内嵌代码片段

### 27_MultiMaskAttentionAggregation/mmagg_kernel.h.snippet.h

```cpp
        SyncM3V_(pipe_);
        absC_[e] += k;
        pendC_[e] -= k;
        if (pendC_[e] > 0) {
            for (int64_t i = 0; i < pendC_[e]; ++i) {
                stage.SetValue(static_cast<int32_t>(i),
                               stage.GetValue(static_cast<int32_t>(k + i)));
            }
            SyncSV_(pipe_);
        }
    }

    // ================= legacy path (P % Q != 0) — verified, untouched =====
    __aicore__ inline void ProcessRun_(int64_t c, int64_t pa, int64_t pb)
    {
        curClass_ = c;
        SyncVM3_(pipe_);
        DataCopy(outGm_[stageAbs_], stage, static_cast<int32_t>(k));
        SyncM3V_(pipe_);
        stageAbs_ += k;
        pending_ -= k;
        if (pending_ > 0) {
            for (int64_t i = 0; i < pending_; ++i) {
                stage.SetValue(static_cast<int32_t>(i),
                               stage.GetValue(static_cast<int32_t>(k + i)));
            }
            SyncSV_(pipe_);
        }
    }
```
