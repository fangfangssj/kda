---
type: Runbook
title: "Multi-launch fused pipeline: carry inter-stage GM intermediates in fp32, round once at the final epilogue"
description: "Multi-launch fused pipeline: carry inter-stage GM intermediates in fp32, round once at the final epilogue"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-696"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-696 — OL-303: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend910_9382 (arch 220x); cann=9.0.0; bisheng=n/a; op_class=fused multi-stage pipelines staged through GM (cube→vec→cube, mm→act→mm, chained matmul)。
- 验证来源：soc=Ascend910_9382; cann=9.0.0 (2_FFN canonical eval, 2026-07-25)。

**OL-696** — multi-launch fused pipeline: carry inter-stage GM intermediates in fp32, round once at the final epilogue. fp16/bf16 storage between stages double-rounds (post-mm1 + post-act) vs the fused vendor kernel's fp32-L0C-through-activation; measured eco_mere 2e-3..3.7e-1 + opposite-sign-Inf on fp16-intermediate vs 4.46e-6 fp32-intermediate control. Middle design (fp16 cube + fp32 act + single fp16 act-out) = vendor-parity precision but strict-≤ T2 coin flip. (2_FFN 2026-07-25, arch 220x)

- Former A5Ops ID: `OL-303`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend910_9382 (arch 220x); cann=9.0.0; bisheng=n/a; op_class=fused multi-stage pipelines staged through GM (cube→vec→cube, mm→act→mm, chained matmul)`
`verified_on: soc=Ascend910_9382; cann=9.0.0 (2_FFN canonical eval, 2026-07-25)`
`unverified_on: soc=Ascend950PR — the rounding principle is dtype math and should transfer, but the cost balance (fp32 cube ≈ half-rate + 2× intermediate GM traffic) is chip-specific`

**Principle**: every store of an intermediate to GM in a low-precision dtype rounds. A pipeline `f16 mm1 → f16 act → f16 mm2` rounds TWICE before the final output (post-mm1 accumulator, post-act), while the fused vendor kernel keeps the fp32 L0C accumulator through the activation and rounds only at mm2's A-side. Measured vs an fp32 CPU-truth golden (2_FFN, 14 fp16 inner-precise cases): the fp16-intermediate pipeline sat at eco_mere 2e-3..3.7e-1 and produced **opposite-sign-Inf** at degenerate draws (fp16 intermediate rounding flips the sign of ±1e5-cancellation sums → hard FAIL with 0/0 metrics), while the fp32-intermediate control sat at mere 4.46e-6.

**Design rule (applies to ANY GM-staged fused pipeline)**:
1. Route inter-stage GM buffers in fp32 (`f16f32` cube variant + fp32 activation + downcast only at the final epilogue) — one rounding total, matching the fused vendor kernel's effective precision.
2. Measure the perf tradeoff (fp32 cube ≈ half rate + 2× intermediate GM traffic) against the perf floor BEFORE committing; the middle option (fp16 cube at full rate + fp32 act + a single fp16 rounding at act-out) reaches vendor-parity precision but makes a strict-≤ T2 comparison vs the vendor kernel a coin flip.

**Evidence**: 2_FFN (2026-07-25, 58 cases): fp16-intermediate pipeline = 14 pass-A cases at eco_mere 2e-3..3.7e-1 + 11 opposite-sign-Inf cases; switching fp16 ip=1 routing to the fp32 pipeline took the canonical grade from 41/58 to 55/58 (36 T1 + 19 T2, 3 degenerate-draw skips).

**Other instances (predicted)**: any chained-GEMM fused op (MLP, gated-FFN, attention QK^T→softmax→PV staged through GM, conv chains); any multi-launch pipeline whose vendor reference is a single fused kernel.

**Cross-ref**: CAND-A5OPS-0012 (the oracle-dtype design that made this loss visible — a low-precision CPU oracle had masked it), CAND-FFN-STAGED-CUBE-PIPELINE (the pipeline architecture this applies to), CAND-FA-MULTI-LAUNCH-PERF-GAP delta #1 (the perf side of the same GM-round-trip coin). backend=ascendc.
