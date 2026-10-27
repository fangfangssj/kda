---
type: Runbook
title: "OBSERVED — an fp16 intermediate in a `dequant → SwiGLU → dynamic-quant` chain overflows on the **post-activation product `silu(a) * b`** computed in fp16"
description: "OBSERVED — an fp16 intermediate in a `dequant → SwiGLU → dynamic-quant` chain overflows on the **post-activation product `silu(a) * b`** computed in fp16"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-76"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-76 — PB-76: OBSERVED — an fp16 intermediate in a `dequant → SwiGLU → dynamic-quant` chain overflows on the **post-activation

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579；CANN 9.1.0.B150；op_class=grouped_matmul + swiglu + per-token int8 quant。
- 验证来源：a5_ops 4_GroupedMatmulSwigluQuant（2026-09-07）；经非作者子代理独立复推。

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150; op_class=grouped_matmul + swiglu + per-token int8 quant`
`verified_on: a5_ops 4_GroupedMatmulSwigluQuant, 2026-09-07; independently re-derived by a non-author sub-agent`
`mechanism: CONFIRMED by direct magnitude measurement over all 60 frozen cases`

A kernel accumulated the cube result exactly in `int32`, dequantised in `fp32`, then cast to `half`
before the activation chain:

    Cast(fp16Full, fp32Full, RoundMode::CAST_RINT, n_);
    Sigmoid(sigmoid, fp16Full, halfN_);
    Mul(sigmoid, sigmoid, fp16Full, halfN_);          // silu
    Mul(fp16Full, sigmoid, fp16Full[halfN_], halfN_); // post = silu(a) * b

Every quantised output became **zero**, on every case.

⚠️ **The overflowing value is NOT the cast input.** Measured over the 60 frozen cases: the
dequantised matmul values peak at only **~5.6e3**, comfortably inside fp16. It is the
**post-activation product `silu(a) * b`** that reaches **1.68e7**, computed *in* fp16 — and
**100% of rows in every case** exceed fp16's 65504 ceiling. The row absmax becomes Inf, the
quantisation scale becomes Inf, and the int8 result collapses to zero.

**Why this is dangerous beyond the wrong answer:** if the *reference* performs the same fp16 step,
reference and candidate degenerate together and the precision gate stops discriminating — a NULL
candidate returning only correctly-shaped zeros and Infs then passes the full denominator. Diagnose
with a non-degenerate CPU truth (see P-P250), not with the gate.

**Fix**: keep the activation chain in fp32 (`Sigmoid`/`Mul`/`Mul` directly on the fp32 buffer,
widening the activation scratch from `half` to `float`). Measured effect of this change alone: from
all-zero output to bit-exact on 58/60 cases against a CPU truth built on the frozen input semantics.
The later 60/60 under the current-contract verifier belongs to a different arm that also applied the
per-token input scale; under that contract the fp32-activation arm without the input scale scored
y 0/12 on a 12-case subset — see P-P250.

**Whether the chain overflows depends on the input scales — measured only indirectly.** The 1.68e7
above comes from an input generation drawing both scales from `rand()` -> [0, 1). The operator's
current source draws them from `rand()*0.049 + 0.001` -> [0.001, 0.05], ~20x smaller. Under that
generation an fp16-activation arm and an fp32-activation arm (both without the per-token input
scale, 12 cases) both recorded a finite scale error of ~1.6e6, so the fp16 chain did not overflow there. Not
measured: `max|post|` under the new scales, and gate discrimination against an fp16 chain (it was
measured against an fp32 reference). So treat this as a **data-scale-dependent** failure: the
identical kernel can be fine or fatal depending on the scale distribution its inputs are drawn
from. This is the concrete reason to measure `max|post|` rather than reason about dtypes.

**Related, not the same defect**: the operator's original producer (Triton / 910b2) recorded an fp16
*precision* miss on this chain and applied the same fp32 remedy — "Accuracy failure (60/60 failed)
with fp16 intermediate (mm_out/swig_tmp fp16): out_s relative error ~4e-4, exceeding fp32 tolerance
1.22e-4. Fix: changed mm_out and swig_tmp to torch.float32." A finite ~4e-4 relative error is fp16
rounding, not overflow: it supports the remedy, not the overflow mechanism.
[OL-227](../precision/ol_227_gated_backward_fp16_overflow_dtype_range_edge.md) records, in backward, the same activation family's output
gate amplifying the upstream gradient until it overflows fp16 first.

⚠️ **Unresolved tension, flagged and not resolved here**: canonical P-P104
(`patterns/domains/gmm_swiglu_quant_a8w8_class_template.md`, same op class and SoC) sets
`dequantDtype = 1 (fp16/half)` in its W4 configuration and dequantises in fp16. Whether a chain built
that way overflows depends on the input scales, as above. Measure `max|post|` on the target's inputs
before following either entry.

**What is NOT claimed**: no statement about other operators, other SoCs, or fp16 activation chains
whose post-activation magnitude stays below 65504. The threshold is a property of the data, not of
the op — measure `max|post|` before concluding.
