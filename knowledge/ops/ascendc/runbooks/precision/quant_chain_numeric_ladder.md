---
type: Runbook
title: "Numeric ladder for matmul → dequant → activation → dynamic quant"
description: "量化链数值位宽阶梯：int8×int8→int32 累加→fp32 dequant→fp32 激活→int8 量化；按各阶段输出幅值选位宽——激活前转 fp16 看似安全，激活后乘积可溢出 fp16（4-4 实测 silu(a)*b 峰值 1.68e7）。"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P250"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# quant_chain_numeric_ladder

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B150（卡内 applies_to 行）；domain=quant_chain；trigger 为任何以 per-token/per-row 动态量化收尾且前置非线性激活的算子（卡内 trigger 行）。
- 验证来源：a5_ops 4_GroupedMatmulSwigluQuant（Ascend950PR, CANN 9.1.0.B150）2026-09-07，由非作者子代理独立复推；同一 ladder 见于同族算子 A3→A5 移植档案（y_int8 BIT_EXACT）（卡内 verified_on 行）。未验证范围：其他 SoC/CANN 版本——机制为 arch-neutral 的 IEEE fp16 动态范围（max 65504），但 ladder 数值仅在上述平台实测，他处须重测 max|post|（卡内 unverified_on 行）。


## P-P250: Numeric ladder for `matmul → dequant → activation → dynamic quant`

    applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150
    unverified_on: every other SoC / CANN version. The mechanism is IEEE fp16 dynamic range
               (max 65504), which is arch-neutral, but the ladder numbers were MEASURED on
               Ascend950PR_9579 / CANN 9.1.0.B150 only (see verified_on); re-measure max|post| elsewhere.
    domain: quant_chain
    trigger: any op ending in a per-token/per-row dynamic quantisation after a non-linear activation
    severity: HIGH
    verified_on: a5_ops 4_GroupedMatmulSwigluQuant (Ascend950PR, CANN 9.1.0.B150), 2026-09-07,
                 independently re-derived by a non-author sub-agent; the same ladder appears in the
                 in-tree A3→A5 port of the sibling operator, whose archive reports y_int8 BIT_EXACT.
    id: P-P250 (owner block main). Drafted as F-P6 on an unmerged branch; never canonical under that id.

### The pattern

Carry each stage at the width its **output magnitude** needs, not the width its input had:

    int8 x int8   ->  int32 accumulate      exact: |sum_K| <= 128*128*K stays inside int32
    int32         ->  fp32 dequant          multiply by per-channel and per-token scales in fp32
    fp32          ->  fp32 activation       silu / sigmoid / gate product ALL in fp32
    fp32          ->  int8 quant            scale = absmax * (1/127); round; clamp; cast

The load-bearing step is the third. A cast to fp16 *before* the activation looks safe because the
dequantised values are small, but the **product after the activation** can be orders of magnitude
larger. On 4-4 the dequant values peaked at ~5.6e3 while `silu(a)*b` reached 1.68e7 — see
PB-76（A5Ops 绑定外部引用；本语料未收录目标）. OL-227（A5Ops 绑定外部引用；本语料未收录目标） records, in
backward, the same activation family's output gate amplifying the upstream gradient until it
overflows fp16 first.

⚠️ **Unresolved tension, flagged and not resolved here**: canonical P-P104
(`gmm_swiglu_quant_a8w8_class_template.md`, same op class and SoC) sets `dequantDtype = 1 (fp16/half)`
and dequantises in fp16, the opposite of the third step above. Whether that overflows depends on the
input scales (PB-76). Measure `max|post|` on the target's inputs before following either entry.

### Two semantics that must be measured, never assumed

- **Which half is the gate.** Probe the actual op rather than reading the name: on this stack
  `npu_swiglu` computes `silu(first_half) * second_half` (author probe: max error 1e-8, against 1.73
  for the other convention; a separate verifier probe: 5.96e-8, against 0.68).
- **Whether a per-token input scale is applied at all.** It does **not** cancel under per-row absmax
  normalisation, because the activation is non-linear: scaling both halves by `s` yields
  `s^2 * gate * up * sigmoid(s*gate)` — the sigmoid argument moves. Measured without it: a truth
  built without the scale matched on 0/60 cases (max|dy| up to 127), and a candidate without it
  scored 0/12 on a 12-case subset (max|dy| 21-85). With it, the candidate reached 38/60 bit-exact,
  the rest at |dy| = 1.

### Building a truth that can actually discriminate

Compute the golden on CPU with the ladder above (int32 matmul via `.to(torch.int32)`, everything
after in fp32). Then prove the truth discriminates before trusting a pass: run a **known-bad**
variant through it. Two controls that must collapse it — the wrong gate convention, and the per-token
scale wrongly applied or omitted. On 4-4 both drove bit-exact agreement from 58/60 to 0/N.

⚠️ Use the **bit-exact count** as the discriminator, not mean elementwise agreement: under wrong
configurations the mean stayed above 96% in one measurement, and rounded to 1.0000 on near-misses in
another. It is unreliable in both directions.

### Not claimed

The ladder is a width-selection rule, not a correctness proof. It does not address rounding-order
differences between CPU and device, which remain at a 1-2 ULP floor on the transcendental, nor does it
say anything about performance — fp32 activation is more expensive than fp16 and that cost is
unmeasured here.
