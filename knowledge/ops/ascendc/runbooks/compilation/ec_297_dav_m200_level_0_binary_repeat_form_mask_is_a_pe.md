---
type: Runbook
title: "dav-m200 Level-0 binary repeat-form `mask` is a PER-REPEAT element count — fp32 silently clamps at 64; long axes must be chunked or use the Level-2 contiguous form"
description: "mask/broadcast 二元调用 `calCount > 64`(fp32) 时每个 repeat 窗口只算前 64 个元素，静默部分计算，无 trap。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-297"]
---

# dav-m200 Level-0 binary repeat-form `mask` is a PER-REPEAT element count — fp32 silently clamps at 64; long axes must be chunked or use the Level-2 contiguous form

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any VEC binary/unary call in the (dst, src0, src1, count, repeatTime, RepeatParams) form)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (27_MultiMaskAttentionAggregation kw-1 respawn, 2026-09-04)`
`unverified_on: soc=Ascend950PR (351x) / Ascend910 (220x) — the clamp bound is lanes-per-repeat × blocks-per-repeat and was not measured on other SoCs`
- **Symptom**: silent partial computation — no trap, no compile error. A masked/broadcast binary call such as `Mul(dst, src0, src1, calCount, repeatTime, BinaryRepeatParams(...))` with `calCount > 64` (fp32) computes only the FIRST 64 elements of every repeat's window. Downstream reductions then see a prefix of each row: an all-ones-mask probe returned `num[v] == sum_{n<64} p[v][n]` EXACTLY (prefixK=64 on every row). Shapes with N ≤ 64 PASS; every N > 64 case fails with a uniform-scale signature (a fraction N/64 of the masked mass missing) or, after per-shape amplification, garbage output.
- **Root cause**: in the Level-0 MASK-COUNT overload `Op(dst, src0, src1, uint64 mask, uint8 repeatTime, BinaryRepeatParams)` (overload identity per CANN 安装树内 kernel_operator_vec_binary_intf_impl.h) the 4th argument is the PER-REPEAT element count, NOT a total. On dav-m200 a `vmul` repeat spans 8 blocks × 8 fp32 lanes = 64 elements; a count above that lane budget is silently clamped to the per-repeat window.
- **Fix**: chunk the long axis into ≤64-element segments per call (advance the mask-row base per chunk; `src1RepStride=0` still broadcasts the row operand across repeats), or use the plain Level-2 contiguous form `Op(dst, src0, src1, count)` which auto-loops (validated at count=4096). Anchor (`代码片段 27_MultiMaskAttentionAggregation/mmagg_kernel.h.snippet.h（见文末）`):
  ```cpp
  constexpr int64_t kColChunk = 64; // 8 blocks x 8 fp32 lanes per repeat
  for (int64_t off = 0; off < N_; off += kColChunk) {
      const int64_t cnt = (N_ - off < kColChunk) ? (N_ - off) : kColChunk;
      Mul(prod[off], p[off], maskRow_[off], static_cast<uint64_t>(cnt),
          static_cast<uint8_t>(vact),
          BinaryRepeatParams(1, 1, 1, NP_ / 8, NP_ / 8, 0));
  }
  ```
- **Detection**: all-ones-mask probe — if the masked dot equals the first-64 prefix on every row, this is the bug. Also: a shape split where ALL small-N cases pass and ALL large-N cases fail with scale-fraction errors.
- **Not a contradiction of OL-51**: OL-51 encourages the HIGH-LEVEL counter mode, where the count IS a total and hardware auto-loops main+tail. The trap is only the LOW-LEVEL 6-arg repeat form, whose 4th arg merely looks like a total.
- **Evidence**: 27_MultiMaskAttentionAggregation 310p kw-1 respawn (2026-09-04): probe pinned prefixK=64 exactly; after chunking, fp32 errors dropped from ~5e-3/inf to ~1e-9 across the N=64..512 sweep; full op Pass A 50/50 T1, Pass B 93/93 T1, det 50/50 bit-exact.
- **Other instances (predicted)**: masked softmax denominators, gated matmul epilogues, boolean-mask reductions — any port that keeps a repeat-form call and lets the count scale with row width N.
- **Cross-ref**: OL-51 (the safe high-level total-count counter form), [`EC-294`](ec_294_dav_m200_setflag_waitflag_hardevent_mte2_s_deadl.md) (dav-m200 VEC-semantics family). backend=ascendc.

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

## 适用场景

- **工程形态**：mask/broadcast 二元调用 `calCount > 64`(fp32) 时每个 repeat 窗口只算前 64 个元素。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
