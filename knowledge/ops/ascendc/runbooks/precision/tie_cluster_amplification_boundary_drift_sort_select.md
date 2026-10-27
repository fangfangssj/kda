---
type: Runbook
title: "Tie-cluster amplification of OL-83 boundary drift in low-precision sort+select pipelines (1-op evidence)"
description: "Tie-cluster amplification of OL-83 boundary drift in low-precision sort+select pipelines (1-op evidence)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP79"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP79 — Tie-cluster amplification of OL-83 boundary drift in low-precision sort+select pipelines (1-op evidence)

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；场景为 bf16/fp16 大 N（≥16384）sort+select 管线中 tie 簇放大 OL-83 边界漂移（正文 Pattern 段）。
- 验证来源：卡内无 verified_on 行；来源为 op#9 9_TopKTopP pp-3（2026-05-03）探针输出（pp3_t1_vs_cpu_diff.json + pp3_tie 分析，卡内 Source 行），症状签名见正文 Symptom signature 段。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Pattern**: When a sort+top-K+top-P (or sort+threshold-mask) op runs on `bf16` or `fp16`
with large N (≥16384), random-Gaussian inputs produce dense tie clusters at the top-K
boundary. The limited mantissa quantizes ~N distinct fp32 values down to ~2^M (M=mantissa
bits) distinct bit patterns; with N >> 2^M the boundary lands inside a 100s-position tie
cluster. Different valid sort implementations break ties differently → different valid top-K
masks → different top-P walks → different valid emit positions. Verifier reports
`max_abs_diff = FLT_MAX` and `mean_abs_diff = inf`, but `finite_max_diff = 0.0`.

**Symptom signature**:
- Pass A (small-N edge-set, deterministic shapes) bit-exact ✓
- Pass B (random benchmark, N≥16K, bf16/fp16) shows `max_abs_diff = 3.4e38`, `mean_abs_diff = inf`
- BUT `finite_max_diff = 0.0` (only -inf vs finite mask flips, no value disagreement)
- T1-vs-CPU triage (CAND-PP80) shows `kernel_vs_cpu_truth_flips ≈ ref_vs_cpu_truth_flips`
  (vendor reference is NOT MORE CORRECT than our kernel vs fp64 truth)

**Mitigation paths (decision rule)**:
- **DO NOT** attempt to reproduce vendor's tie-break order — vendor tie-break depends on
  hardware-internal sort intrinsic (undocumented, version-specific). Reproducing is OL-85
  case-specific overfitting that breaks on vendor updates.
- **PREFERRED**: classify cluster as OL-83-amplified T2-with-evidence carry-over.
- **ROOT FIX (methodology layer)**: refine verifier to admit
  `n_flip_positions ≤ Σ_rows tie_count_at_kth_value` AS T2-with-evidence pass, conditional
  on `finite_max_diff = 0.0`. This is a DEBT-level methodology improvement (not per-op).

**Concrete anchor — bf16 N=65536 random Gaussian**:
```
~256 distinct values in [-3σ, +3σ] → ~256 ties per unique value
Random k in distribution bulk → boundary lands inside 248-284-position tie cluster
Our Sort<MERGE_SORT> keeps 121-of-248 ties; CANN ref keeps 74-of-248 — both spec-valid
```

**Evidence**:
- op#9 9_TopKTopP cluster {8, 17, 26, 35} on bf16 [B, 65536] (2026-05-03, pp-3):
  `finite_max_diff=0.0` everywhere; `kernel_vs_cpu_truth` flips ≈ `ref_vs_cpu_truth` flips.

**Other instances (predicted)**:
- Any op that does sort + boundary-threshold-emit on bf16/fp16 large-N: top-k softmax
  (op#7 MoeGatingTopKSoftmax has potential exposure on dense small-num_experts cases),
  threshold-mask scatter, NMS at low IoU thresholds with score ties, beam search.
- Generalizes to any "rank-then-cut" op family.

**Promote when**: a 2nd op exhibits the same `finite_max_diff = 0.0 ∧ flips ≈ ref_flips`
T1-vs-CPU triage signature in a non-9_TopKTopP-family op.

**Source**: op#9 9_TopKTopP pp-3 (2026-05-03), workspace/9_topktopp/probes/probe_outputs/pp3_t1_vs_cpu_diff.json + pp3_tie_analysis.txt.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP79，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
