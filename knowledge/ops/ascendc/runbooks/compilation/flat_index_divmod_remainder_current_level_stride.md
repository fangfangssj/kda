---
type: Runbook
title: "flat-index → multi-dim divmod chain: each level's remainder must use that level's stride, not the innermost"
description: "flat-index → multi-dim divmod chain: each level's remainder must use that level's stride, not the innermost"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["OutputOffsetToInputIndex"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-195 — EC-195: A5Ops additive extension

> 关联：同一 EC-195 主题已由既有卡 [ec_195_flat_index_multi_dim_divmod_chain_each_level_s_r.md](ec_195_flat_index_multi_dim_divmod_chain_each_level_s_r.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（纯索引算术逻辑 bug，与 arch 无关）；CANN=all；op_class=pooling 及任何将多维输出索引线性化后再分解回各维坐标的算子。
- 验证来源：卡内 verified_on 声明为 n/a（算法/逻辑正确性 bug，可在源码层验证）；实测证据见正文 Evidence 行（adaptive_avg_pool3d 310p，2026-08-10，iter-2 修复后 12/12 PASS）。

**EC-195** — flat-index → multi-dim divmod chain: each level's remainder must use THAT level's stride, NOT the innermost (=1, so `rem % 1 == 0` collapses the coordinate to 0). Symptom: uniform whole-output error (one dim always reads 0), invariant to dtype/cast tweaks. Reuse the upstream closed form (`ow = offset % hstride / wstride`) instead of re-decomposing. Arch-independent logic bug. Anchor 内部路径（不公开）workspace/adaptive_avg_pool3d/kernel/adaptive_avg_pool3d_kernels.cpp:191-195, 12/12 PASS after iter-2 fix. Cross-ref [OL-160](ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

- Former A5Ops ID: `EC-195`
- Fusion disposition: `additive_new`

`applies_to: soc=all (pure-index-arithmetic logic bug — arch-independent); cann=all; bisheng=n/a; op_class=pooling / any op that linearizes a multi-dim output index and decomposes it back to per-dim coordinates`
`applies_to_backend: ascendc`
`verified_on: n/a — this is an algorithm/logic-correctness bug (source-verifiable in the kernel), not a device-runtime claim; the principle is arch-independent.`

- **Category**: precision (logic)
- **Loaded by**: Generator/Worker when a kernel iterates a FLAT output index and decomposes it into per-dimension coordinates (n, d0, d1, …, ow) to map an output point back to its input region — adaptive pooling, strided gather/scatter, multi-dim unary emit, coordinate-grid kernels.
- **Symptom**: large, UNIFORM precision error across the whole output (e.g. `max_abs_diff` 0.5–4.4 on every case) that does NOT shrink with dtype/cast-mode tweaks; only the first slice along one dimension is correct, and every point with a non-zero coordinate in that dimension reads as if the coordinate were 0.
- **Mechanism**: the standard divmod chain to split a flat index `f` into per-dim coords (with `stride_i` = product of all inner dims below dim `i`, innermost stride = 1) is `coord_i = rem / stride_i; rem = rem % stride_i`, processed OUTER→INNER. The bug: taking the INNERMOST level's remainder for an outer/middle level. Because the innermost stride is 1, `rem % 1 == 0` always, collapsing that coordinate to 0 for every point. The correct remainder at level `i` uses `stride_i` (the CURRENT level's stride), not the final level's stride.
- **Fix**: at each level of the divmod chain, modulus by the current dimension's stride, then carry the remainder forward. Equivalently, reuse the upstream closed form when one exists — CANN `OutputOffsetToInputIndex` computes `ow = offset % hstride / wstride` (where `hstride = outW`, `wstride = 1`), i.e. it already folds the inner two levels correctly via the trailing `/ wstride`; do NOT re-decompose `ow` with `% wstride`.
- **Correct pattern** (anchor, 内部路径（不公开）workspace/adaptive_avg_pool3d/kernel/adaptive_avg_pool3d_kernels.cpp:191-195 — mapping a flat output spatial point to (od, oh, ow); 内部路径（不公开）adaptive_avg_pool3d_kernel.h:24-31 defines `dstride=H*W, hstride=W, wstride=1`):
  ```cpp
  // WRONG (iter-1 bug): ow = rem % wstride;   // wstride == 1 -> ow ALWAYS 0
  // RIGHT (shipped):     ow = rem % hstride;   // hstride == outW, the CURRENT level's stride
  int64_t rem = outputPointIdx % outputShape.nstride;
  int64_t od  = rem / outputShape.dstride; rem = rem % outputShape.dstride; // dstride = H*W
  int64_t oh  = rem / outputShape.hstride;                                     // hstride = W
  int64_t ow  = rem % outputShape.hstride;                                     // current-level stride, NOT wstride(=1)
  ```
- **Evidence**: adaptive_avg_pool3d 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 produced `max_abs_diff` 0.5–4.4 across ALL 12 cases (9 fp32 + 3 fp16): every output point with `ow > 0` computed as if `ow = 0`. Root cause was `rem % wstride` (innermost stride ⇒ always 0) instead of `rem % hstride`. Fixed in iter 2 by matching the upstream `OutputOffsetToInputIndex` form (内部路径（不公开）adaptive_avg_pool3d_kernel.h:55-60); 12/12 PASS. Logged `novel` in `workspace/adaptive_avg_pool3d/failures_ledger.md` iter 1. Device provenance UNVERIFIED (`hardware/target/ascend310p.md` `device.runtime: NOT_RUN`) — but this is a LOGIC bug (source-verifiable: the wrong-stride expression was visible in the kernel), NOT a device-runtime claim, so the provenance flag does not block it.
- **Other instances (predicted)**: adaptive_avg_pool2d, max_pool3d/avg_pool3d with flat output iteration, strided/gathered emit that flattens then re-decomposes, any coordinate-recovery from a linearized ND index. The fingerprint (one dimension always reads 0 + uniform whole-output error, invariant to dtype changes) isolates this bug from a rounding/precision residual.
- **Cross-ref**: OL-160 (canonical entry points the CPU-truth reference runs through — a logic bug of this magnitude fails CPU-truth uniformly across every dtype, distinguishing it from a dtype/rounding residual). backend=ascendc.
