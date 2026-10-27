---
type: Runbook
title: "Hardware reduce intrinsics collapse instruction count, not element work — on dav-m200 the VEC pipe is element-throughput-bound, so swapping manual reduction trees for `vcadd`/`vcmax`/`vcgadd`/`vcgmax` gives zero speedup (probe the pipe's bound before trading instruction count for time)"
description: "dav-m200 的硬件归约 intrinsic（vcadd/vcmax 等）是元素吞吐 bound 而非发射 bound（与 A5 相反）——少指令≠更快"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.compilation, topic.compute.reduction, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-682"]
---

# Hardware reduce intrinsics collapse instruction count, not element work — on dav-m200 the VEC pipe is element-throughput-bound, so swapping manual reduction trees for `vcadd`/`vcmax`/`vcgadd`/`vcgmax` gives zero speedup (probe the pipe's bound before trading instruction count for time)

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=reduction-heavy VEC kernels (softmax row-max/sum, norm statistics, attention PV/score reductions, quant epilogue sums)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (joint_attention 310p, 2026-08-29 — perf-floor probe, softmax treeW up to 2048)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the A5 VEC is documented as ISSUE-bound (OL-231/OL-238) where the OPPOSITE holds (fewer instructions = faster); this element-throughput-bound reading is dav-m200-specific and does NOT transfer`
**Principle**: on dav-m200 the VEC reduce intrinsics — `vcadd` (WholeReduceSum), `vcmax` (WholeReduceMax), `vcgadd` (BlockReduceSum), `vcgmax` (BlockReduceMax), plus `vcmin`/`vcgmin`/`vcpadd` — ARE implemented and silicon-correct (CANN 安装树内 dav_m200/kernel_operator_vec_reduce_impl.h maps them; CANN 安装树内 Ascend310P3.ini lists all six for fp32). But they only collapse INSTRUCTION COUNT, not ELEMENT WORK: `VREDUCE_PER_REP_OUTPUT=2` means each hw-reduce collapses a 256B (64 fp32) repeat to 2 outputs, so the Level-2 count-form `ReduceSum/ReduceMax(dst, src, tmp, count)` is still a 3-step tree, not a single instruction. And the m200 VEC pipe is ELEMENT-THROUGHPUT-bound (not issue-bound), so a 1024-wide sum reads+adds ~1024 elements whether expressed as a 10-level manual halving tree or 3 hw-reduce calls → ZERO speedup.
**The general rule (probe the bound first)**: instruction-count reduction is a perf lever ONLY when the executing pipe is issue/latency-bound (each op costs the same regardless of width/element count). When the pipe is element-throughput-bound (cost ∝ element count), collapsing N instructions into M does not reduce time — the element reads/adds still dominate. On A5 the VEC is issue-bound (OL-231/OL-238), so the SAME hw-reduce/instruction-count lever CAN help there; on dav-m200 it is inert. The discriminator is the pipe's bound, not the primitive's existence — read the msprof ratio (`aiv_vec_ratio` ≈ 0.85 with tiny `mte2_ratio` ⇒ VEC-bound; then decide whether the VEC is issue-bound or element-bound) before trading manual trees for hw-reduce.
**Sub-finding — Level-2 count-form masking (correctness gotcha, worth keeping)**: `ReduceSum(dst, src, tmp, count)` handles arbitrary-count masking internally (bodyCount=64 + repeatTime=count/64 + tail) — this correctly masks the `-inf` tail of a partially-Exp'd softmax row (sum is NOT tail-safe; max is). The `sharedTmpBuffer` needs only ~2*repeatTime+ε elements, NOT `count` elements.
**Evidence**: joint_attention 310p (2026-08-29, Ascend310P3/dav-m200, CANN 9.0.0): softmax TreeMax/TreeSum → Level-2 count-form probe, treeW up to 2048; ST 220/221 (1 infra flake, re-ran clean); perf_floor 105302us vs 104404us baseline = NO speedup; msprof `aiv_vec_ratio 0.849`, `mte2_ratio 0.109` → VEC-pipe bound, not issue-bound. Corrects two false premises: (1) ko's "no hw-reduce primitive on m200 (only dav_m300/dav_c310)" — FALSE, they exist and run correctly; (2) researcher/fo's "hw-reduce → collapse 40-55% VEC instructions → 104→73-94ms" — FALSE, instruction-count reduction does not translate to time on an element-throughput-bound VEC (`joint_attention__opsmath/knowledge_update.md`; msprof in `joint_attention__opsmath/failures_ledger.md` perf iter 1).
**Other instances (predicted)**: any dav-m200/310P reduction-heavy VEC kernel — softmax row-max/sum, norm statistics, quant per-row scale/offset sums, attention PV/score reductions. The real lever on element-throughput-bound 310P dense VEC-fp32 attention is reducing ELEMENT work (sparsity, algorithm) or accepting the cube's fp32→fp16 precision tradeoff — not instruction count.
**Cross-ref**: OL-52 (reduction-op selection — refine its "small data use WholeReduce for low latency" with this element-throughput-bound caveat), [`OL-416`](../compilation/ol_416_on_dav_m200_wholereducesum_dst_src_mask_repeatti.md) (dav-m200 WholeReduceSum mask = element count — same reduce-primitive family), [`OL-231`](ol_231_unoptimized_prototype_not_architecture_verdict.md)/[`OL-238`](../compilation/ol_238_a_finalize_phase_o5_mismatch_pass_claimed_but_not.md) (the A5 issue-bound CONTRAST — instruction-count IS the lever there), [`OL-20`](ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md) (msprof ratio reading). backend=ascendc.

## 适用场景

- **工程形态**：dav-m200 的硬件归约 intrinsic（vcadd/vcmax 等）是元素吞吐 bound 而非发射 bound（与 A5 相反）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
