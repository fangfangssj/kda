---
type: Runbook
title: "dav_3510 has a real fp32 img2colv2_cbuf_to_ca; on a conv-stack op the bias/act epilogue can dominate the vendor gap while im2col has no vendor counterpart"
description: "dav_3510 has a real fp32 img2colv2_cbuf_to_ca; on a conv-stack op the bias/act epilogue can dominate the vendor gap while im2col has no vendor counterpart"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-422"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-422 — OL-422: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=conv-stack-attention; scope=fused-optimizer-analysis。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0.B060。

**OL-422** — `dav_3510` has a real fp32 `img2colv2_cbuf_to_ca`; on a conv-stack op the bias/act epilogue can dominate the vendor gap (9.57x) while im2col has no vendor counterpart

- Former A5Ops ID: `OL-422`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=conv-stack-attention; scope=fused-optimizer-analysis`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`
`unverified_on: other SoCs; the fusion itself was NOT implemented or measured — only the gap analysis is verified`

**Trigger**: a fused-optimizer pass on an op whose reference decomposes into conv + bias + activation
sub-ops, where an incremental tuning remit has plateaued and the question is *which* sub-op to attack.

**Principle**: per-sub-op attribution against the vendor's own kernels relocates the bottleneck away
from the obvious one. Launch-count wins can mask a per-sub-op deficit that dominates device time.

**Evidence — ⚠️ MIXED PROVENANCE. Read this before quoting any number below.** The two arms of
every cross-arm ratio in this entry come from *different measurement events*:

- **vendor / reference arm — FRESH this pass**: 4 cases, on-device, per-kernel attribution via
  `torch_npu.profiler` Level1 → `kernel_details.csv`. The vendor Relu figure **6.06 us is fresh**.
- **ours arm — NOT fresh; prior run, identity-pinned reuse**: **58.0 / 66.2 / 333.1 us** are
  `bam-ko-2`'s Opt4 profiling (`msprof_opt_4.json`), *not* re-profiled this pass. The bound
  container was bare (no `current_task`, no staged `.so`), so re-profiling our arm would have
  required a full deploy+build. Identity is *pinned, not assumed*: `kernel_md5 0aea5598…` matches
  the on-disk kernel across three independent records.
- **what licenses mixing the arms**: a cross-container reproducibility check — fresh reference
  total **84.01** us/iter vs bam-ko-2's `ref_kernel_sum_us` **84.6** recorded in a *different*
  container (torch_npu 2.7.1) ⇒ **0.7 % agreement**. Without that check the comparison would be a
  cross-container guess.

⇒ **`9.57x` is a mixed-source DIAGNOSTIC (prior-ours 58.0 ÷ fresh-vendor 6.06). It is NOT a
complete fresh A/B, and NOT a product verdict.** Durable source 内部路径（不公开）a5_ops_op36b/workspace/36_BAM/fused_analysis.md:131-145, which
records the reuse verbatim: *"our-arm per-kernel times are reused-but-identity-pinned, not fresh."*
A per-*invocation* breakdown (which of the 6 gemms dominates) was **NOT obtained** and remains the
single largest gap in the analysis.

**Findings that do NOT depend on the mixed arms:**

- hardware im2col `img2colv2_cbuf_to_ca` is **real on `dav_3510` and supports fp32** — confirmed
  against the API catalog, not assumed, and independent of any timing arm.
- the im2col kernel (2 launches, 66.2 us, prior-ours) has **no vendor counterpart at all** — the
  vendor fuses it away. This is **pure added work, not a ratio**: it has no denominator, and must
  never be folded into an average or quoted as a speedup factor.
- im2col + fill0 + bias_act ≈ **124.2 us of 333.1 us (~37 %)** — both terms are prior-ours, so this
  *share* is internally consistent even though the cross-arm 9.57x is not.
- `MatmulImplBase::SetBias(GlobalTensor)` / `(LocalTensor)` exists (内部路径（不公开）matmul_impl_base.h:106,118, CANN toolkit header),
  and the cube currently writes C to GM for a separate AIV launch to read back ⇒ the epilogue is
  foldable **in principle**. ⚠️ **The fusion was NOT implemented and NOT measured.** No speedup is
  claimed here — only that a target was identified and ranked.

**Falsifier**: a fresh BOTH-ARMS A/B on one binary in one container supersedes the 9.57x figure
outright; if it lands near 1x the ranking here is refuted. Otherwise: re-run per-kernel attribution and compare the bias/act kernel against the vendor
activation kernel. If the ratio is near 1x, or if another sub-op exceeds it, this entry's ranking is
refuted for that op. If `img2colv2_cbuf_to_ca` rejects fp32 on the target SoC, the hardware claim is
refuted.

**Rule**: on a conv-stack op, attribute per sub-op against the vendor's own kernels before choosing a
target. Rank by absolute device time AND by vendor ratio; a sub-op the vendor has no kernel for is
added work, not a ratio, and must be reported as such rather than folded into an average.

---