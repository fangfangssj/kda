---
type: Runbook
title: "flat-index → multi-dim divmod chain: each level's remainder must use THAT level's stride, NOT the innermost (="
description: "扁平索引→多维 divmod 拆解时，每层的余数必须用**当层** stride 除，用最内层 stride（=1）会把坐标恒坍缩成 0——整输出均匀错且只有首片正确。"
tags: [task.debugging, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-195"]
---

# flat-index → multi-dim divmod chain: each level's remainder must use THAT level's stride, NOT the innermost (=

**EC-195** — flat-index → multi-dim divmod chain: each level's remainder must use THAT level's stride, NOT the innermost (=1, so `rem % 1 == 0` collapses the coordinate to 0). Symptom: uniform whole-output error (one dim always reads 0), invariant to dtype/cast tweaks. Reuse the upstream closed form (`ow = offset % hstride / wstride`) instead of re-decomposing. Arch-independent logic bug. Anchor `代码片段 adaptive_avg_pool3d/adaptive_avg_pool3d_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]`, 12/12 PASS after iter-2 fix. Cross-ref [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

`applies_to: soc=all (pure-index-arithmetic logic bug — arch-independent); cann=all; bisheng=n/a; op_class=pooling / any op that linearizes a multi-dim output index and decomposes it back to per-dim coordinates`
`applies_to_backend: ascendc`
`verified_on: n/a — this is an algorithm/logic-correctness bug (source-verifiable in the kernel), not a device-runtime claim; the principle is arch-independent.`

- **Category**: precision (logic)
- **Loaded by**: Generator/Worker when a kernel iterates a FLAT output index and decomposes it into per-dimension coordinates (n, d0, d1, …, ow) to map an output point back to its input region — adaptive pooling, strided gather/scatter, multi-dim unary emit, coordinate-grid kernels.
- **Symptom**: large, UNIFORM precision error across the whole output (e.g. `max_abs_diff` 0.5–4.4 on every case) that does NOT shrink with dtype/cast-mode tweaks; only the first slice along one dimension is correct, and every point with a non-zero coordinate in that dimension reads as if the coordinate were 0.
- **Mechanism**: the standard divmod chain to split a flat index `f` into per-dim coords (with `stride_i` = product of all inner dims below dim `i`, innermost stride = 1) is `coord_i = rem / stride_i; rem = rem % stride_i`, processed OUTER→INNER. The bug: taking the INNERMOST level's remainder for an outer/middle level. Because the innermost stride is 1, `rem % 1 == 0` always, collapsing that coordinate to 0 for every point. The correct remainder at level `i` uses `stride_i` (the CURRENT level's stride), not the final level's stride.
- **Fix**: at each level of the divmod chain, modulus by the current dimension's stride, then carry the remainder forward. Equivalently, reuse the upstream closed form when one exists — CANN `OutputOffsetToInputIndex` computes `ow = offset % hstride / wstride` (where `hstride = outW`, `wstride = 1`), i.e. it already folds the inner two levels correctly via the trailing `/ wstride`; do NOT re-decompose `ow` with `% wstride`.
- **Correct pattern** (anchor, `代码片段 adaptive_avg_pool3d/adaptive_avg_pool3d_kernels.cpp.snippet.cpp（见文末） [snippet-unavailable: source workspace not on this box]` — mapping a flat output spatial point to (od, oh, ow); 内部路径（不公开）adaptive_avg_pool3d_kernel.h:24-31 defines `dstride=H*W, hstride=W, wstride=1`):
  ```cpp
  // WRONG (iter-1 bug): ow = rem % wstride;   // wstride == 1 -> ow ALWAYS 0
  // RIGHT (shipped):     ow = rem % hstride;   // hstride == outW, the CURRENT level's stride
  int64_t rem = outputPointIdx % outputShape.nstride;
  int64_t od  = rem / outputShape.dstride; rem = rem % outputShape.dstride; // dstride = H*W
  int64_t oh  = rem / outputShape.hstride;                                     // hstride = W
  int64_t ow  = rem % outputShape.hstride;                                     // current-level stride, NOT wstride(=1)
  ```
- **Evidence**: adaptive_avg_pool3d 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 produced `max_abs_diff` 0.5–4.4 across ALL 12 cases (9 fp32 + 3 fp16): every output point with `ow > 0` computed as if `ow = 0`. Root cause was `rem % wstride` (innermost stride ⇒ always 0) instead of `rem % hstride`. Fixed in iter 2 by matching the upstream `OutputOffsetToInputIndex` form (内部路径（不公开）adaptive_avg_pool3d_kernel.h:55-60); 12/12 PASS. Logged `novel` in `adaptive_avg_pool3d/failures_ledger.md` iter 1. Device provenance UNVERIFIED ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`) — but this is a LOGIC bug (source-verifiable: the wrong-stride expression was visible in the kernel), NOT a device-runtime claim, so the provenance flag does not block it.
- **Other instances (predicted)**: adaptive_avg_pool2d, max_pool3d/avg_pool3d with flat output iteration, strided/gathered emit that flattens then re-decomposes, any coordinate-recovery from a linearized ND index. The fingerprint (one dimension always reads 0 + uniform whole-output error, invariant to dtype changes) isolates this bug from a rounding/precision residual.
- **Cross-ref**: [`OL-160`](ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical entry points the CPU-truth reference runs through — a logic bug of this magnitude fails CPU-truth uniformly across every dtype, distinguishing it from a dtype/rounding residual). backend=ascendc.

## 内嵌代码片段

### adaptive_avg_pool3d/adaptive_avg_pool3d_kernels.cpp.snippet.cpp

```cpp
    {
        // Decompose flat output index into (n, od, oh, ow)
        int64_t n = outputPointIdx / outputShape.nstride;
        int64_t rem = outputPointIdx % outputShape.nstride;
        int64_t od = rem / outputShape.dstride;
        rem = rem % outputShape.dstride;
        int64_t oh = rem / outputShape.hstride;
        int64_t ow = rem % outputShape.hstride;

        // Compute adaptive window bounds
        Index index;
```

**孪生卡 ec-264 合并的独有内容**（ec-264-flat-index-multi-dim-divmod-decomposition-each-l.md 已删除）：

> - **Correct pattern** (anchor, `target/ascendc/examples/adaptive_avg_pool3d/snippets.md` — mapping a flat output spatial point to (od, oh, ow); 内部路径（不公开）adaptive_avg_pool3d_kernel.h:24-31 defines `dstride=H*W, hstride=W, wstride=1`):

> // WRONG (iter-1 bug): ow = rem % wstride; // wstride == 1 -> ow ALWAYS 0

> // RIGHT (shipped): ow = rem % hstride; // hstride == outW, the CURRENT level's stride

> int64_t od = rem / outputShape.dstride; rem = rem % outputShape.dstride; // dstride = H*W

> int64_t ow = rem % outputShape.hstride; // current-level stride, NOT wstride(=1)

## 适用场景

- **工程形态**：扁平索引→多维 divmod 拆解时，每层的余数必须用**当层** stride 除，用最内层 stride（=1）会把。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
