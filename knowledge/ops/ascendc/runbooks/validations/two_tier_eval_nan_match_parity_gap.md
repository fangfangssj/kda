---
type: Runbook
title: "Two-tier eval NaN-match parity gap — `precision_eval_two_tier.py` mis-classifies saturated cases"
description: "Two-tier eval NaN-match parity gap — `precision_eval_two_tier.py` mis-classifies saturated cases"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP81"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP81 — Two-tier eval NaN-match parity gap — `precision_eval_two_tier.py` mis-classifies saturated cases

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；场景为 two-tier 评估器对 kernel 与 CANN 参考同位置饱和 NaN case 的误分类（正文 Pattern/Symptom 段）。
- 验证来源：卡内无 verified_on 行；来源 op#14 14_AdaptiveInstanceNormalization2DBackward pp-1（2026-05-03，卡内 Source 行，DEBT-070 candidate），op#14 分类翻转实测（22/50 strict T1 → 28/50 effective T2）见正文 Symptom 段。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Pattern**: When kernel and CANN reference both saturate to NaN in the same positions where CPU truth is finite, the current `precision_eval_two_tier.py` `classify_output` function evaluates `pass_t2 = (NaN <= NaN)` which is False in Python (NaN comparisons). Net effect: cases where ours matches CANN bit-for-bit on NaN positions get labeled FAIL despite OL-109 parity-or-better intent (we are not strictly worse than CANN — both saturate).

**Symptom**:
- Verifier reports PARTIAL with N FAIL on bf16 / fp16 outputs that have wide dynamic range or division-by-near-zero structure
- Probe ad-hoc check shows ours == CANN bit-for-bit on NaN/inf positions; both diverge from CPU truth equally
- Op classification flips significantly when NaN-match parity is honored (op#14 went 22/50 strict T1 → 28/50 effective T2)

**Patch candidate** (in `classify_output`):
```python
import math
ours_nan = math.isnan(ours_mere) or math.isnan(ours_mare)
cann_nan = math.isnan(cann_mere) or math.isnan(cann_mare)
if ours_nan and cann_nan:
    verdict = "PASS_T2_NAN_PARITY"
elif pass_t1: ...
elif pass_t2: ...
```

**Evidence**:
- 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03): 10/50 cases (2,3,4,8,9,12,20,28,30,49) all show ours+CANN = NaN/NaN on grad_input + grad_weight, with grad_bias bit-exact 0.0/0.0. Without NaN-match treatment, op verdict PARTIAL with 32 FAIL; with NaN-match treatment, effective 28/50 PASS_T2.

**Other instances (predicted)**: Any fp16/bf16-saturating op evaluated through two-tier — high impact on `adaptive_instance_norm_bwd`, `batchnorm_bwd`, `layernorm_bwd`, any op with `pow(std, -3)` or large-dynamic-range intermediates. Affects all Wave 5 PERF-UNKNOWN re-bench ops with mixed-precision outputs.

**Promote when**: the patch is applied to `precision_eval_two_tier.py`, regression-tested against ≥2 ops with NaN saturation. Until then this is a methodology-fix candidate, not a kernel-coding pattern.

**Source**: op#14 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03). DEBT-070 candidate.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP81，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
