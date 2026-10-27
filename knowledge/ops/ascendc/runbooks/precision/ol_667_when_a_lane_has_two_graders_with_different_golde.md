---
type: Runbook
title: "When a lane has TWO graders with DIFFERENT goldens, measure golden-vs-golden distance per failing case BEFORE kernel iteration — a both-pass intersection smaller than the case set is a structural block requiring an owner decision, not a kernel bug"
description: "双 golden 接受域须按完整谓词证明是否相交；native-as-candidate 失败只是线索，有限坐标绝对界可给出固定合同的冲突证明。"
tags: [task.debugging, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-89-when-the-active-reference-is-non-deterministic-loo.md
    role: supporting
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-90-pytorch-ub-class-ops-with-non-deterministic-refere.md
    role: supporting
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-16T00:00:00Z"
platforms: ["310p", "950"]
aliases: ["OL-667", "native-as-candidate", "dual golden absolute cap", "acceptance regions overlap"]
---

> **来源说明（2026-10-16）**：本卡核心主张的实测证据来自车队私有 run（见 Evidence 节 run_id / verified_on），sources 为同缺陷族公开先例（背景，不承载本卡具体主张）；按社区评审口径降为 draft，待补可复核来源后回升。

# When a lane has TWO graders with DIFFERENT goldens, measure golden-vs-golden distance per failing case BEFORE kernel iteration — a both-pass intersection smaller than the case set is a structural block requiring an owner decision, not a kernel bug

`applies_to: soc=all (observed on Ascend310P3); cann=all (observed 9.1.0); bisheng=n/a; op_class=all (any op graded by ≥2 independent acceptance scripts against different goldens); workflow=verification`
`verified_on: soc=Ascend310P3; cann=9.1.0 (instance_norm_grad 310p, 2026-09-02, benchmark-acceptance lane — measured both-pass intersection 7/41)`

**Principle**: two graders whose goldens differ (e.g. a dtype-native CPU golden on one lane vs an fp64-truth golden on another) impose CONTRADICTORY acceptance regions wherever the golden-vs-golden distance exceeds the sum of their tolerances. No kernel output can satisfy both there; iterating the kernel toward one grader strictly degrades it under the other. A cheap diagnostic probe is **native-as-candidate**: run the real reference golden ITSELF through the other grader's comparator. Failure alone is NOT a proof that the two acceptance regions are disjoint. Prove a conflict from the complete predicates and a bound on at least one fixed coordinate, as below; only the proven fixed-contract conflict requires an owner-level acceptance decision rather than candidate iteration. The probe does not authorize changing input distribution, golden, dtype or grading standard.

**Trigger classifier — verify the TRUTH SOURCE before declaring a structural grader conflict (2nd-instance correction, 2026-09-05)**: the native-as-candidate probe must run the REAL golden-generating program; that provenance is necessary, but a probe FAIL is still not a disjointness proof. A worker-modified `model.py` (not byte-identical to the sha-pinned vendor def) silently re-targets both the kernel and the probe's vendor leg — an apparent golden-vs-kernel contradiction can be an artifact of the modified truth source, not a structural conflict. Check `sha256(model.py) == dataset_provenance.op_def_sha256` (or an explicit diff-justification artifact) BEFORE escalating a both-pass-intersection block to an owner decision; the corrected 2nd-instance evidence below is the catch instance.

**Concrete anchor**: instance_norm_grad 310p — the owner lane grades vs a CPU dtype-native golden (PB-68 class) while the O5 two-tier grader grades vs fp64 CPU truth (normal-region mismatch = instant FAIL). Benchmark inputs carry raw `randn` mean/std (std min ≈ 5e-3), so `pow(std,-3)` (gain ~1e6..1e8) amplifies any noise: the fp16 CPU-native golden cascades inf→NaN while fp64-truncated truth has ±inf and no NaN (18/19 fp16 cases ungradeable by both at once), and fp32 row-stat accumulation-order noise (~1.7e-6 rel — carried by ANY faithful fp32 impl, torch CPU included) amplifies to element rel errors up to 9.9e-2. Measured: the CPU-native golden itself fails the fp64-truth comparator on 12/14 failing fp32 cases; both-pass intersection 7/41.

**Evidence**: instance_norm_grad 310p (2026-09-02): `instance_norm_grad/knowledge_update.md` finding 1 + `instance_norm_grad/failures_ledger.md` precision iter 3; per-case dual-grader evidence table recorded with the op's Phase D artifacts (PROGRESS.md).

**Evidence (2nd instance — CORRECTED 2026-09-05 kw-2; the originally-recorded "structural fp16 golden conflict" is WITHDRAWN)**: 28_MultimodalRopePositionComputationWithGridBasedIndexing 310p (benchmark, DEVICE-RUN device 3) initially read as this entry's 2nd instance — 5 pass_b fp16 "conflicts", "vendor-self fails its own golden", shipped 55/60. The post-worker audit (F1 truth-source swap) found the "vendor" was a WORKER-MODIFIED model.py (sha e0d6c71d) that ADDED dtype-specialized branches the vendored def (sha f195e397, pinned in `dataset_provenance.op_def_sha256`) does not have — the true def accumulates unified-fp32 with ONE final round (in-code marker comment "统一 fp32 累加，消除 fp16/bf16 分支差异"). With model.py restored byte-verbatim and the kernel re-derived from the true contract (exact fp32 weights, per-op fp32 chain, single final RNE round): pass_a 50/50 PASS_T1 with fp32/fp16/bf16 bit-exact (ours_mere=0) on all 6 low-dtype cases AND pass_b 60/60 — the fp64-promoted edge golden was NEVER in conflict with the true vendor semantics. `28_MultimodalRopePositionComputationWithGridBasedIndexing/knowledge_update.md` findings 1-2 + `28_MultimodalRopePositionComputationWithGridBasedIndexing/verification.json` truth_source. Contract-provenance rule: OL-691 (revised).

**Other instances (predicted)**: any op running under parallel acceptance harnesses (owner benchmark scripts + O5 two-tier, vendor-native leg + CPU-truth leg, cross-SoC grading) whose goldens are not bit-compatible — pow/reciprocal/division chains with near-zero denominators are the highest-risk class (P-CAT-6). Generalizes OL-89/OL-90's single-grader truth-regime reasoning to the multi-grader case.

**Cross-ref**: PB-68 (dtype-native truth regime), [`OL-89`](ol_89_when_the_active_reference_is_non_deterministic_loo.md) / [`OL-90`](ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) (Path-A CPU-truth topology), P-P61 (structural-block escalation class), PB-74 (the cast-saturation half of the fp16 divergence), P-CAT-6 (the input-distribution hazard that created the divergence), [`OL-668`](ol_668_a_composite_grader_verdict_ands_independent_sub.md) (reading which sub-check binds). backend=all.

## Finite-coordinate proof and limits (saved Ascend950 observation)

First bind the original input, reference programs/state, output shape/dtype, saved reference bytes and complete comparator. For two fixed finite real-valued references a,b, if each required predicate imposes an all-coordinate absolute bound c_a,c_b, `abs(a-b) > c_a+c_b` proves the permitted intervals do not meet at that coordinate. For a finite-dtype subtraction comparator, derive conservative bounds that include its rounding; do not use a real-arithmetic formula without that check. A matched-ratio allowance or relative-isclose failure alone does not supply an all-coordinate absolute bound. Nonfinite positions need their declared structural predicate. Overlapping intervals are only a necessary condition and do not prove that a candidate satisfies the complete predicate.

Counterexample to the former decisive-probe wording: a=0, b=.015, each absolute cap=.01. Either reference fails the other comparator, yet candidate=.0075 satisfies both. A reference-to-reference FAIL is therefore a discriminator for investigation, not automatically a candidate-independent obstruction.

The new saved observation is one finite FP32 coordinate in a [512,2048] output under a fixed dual-reference contract. The original composite comparator requires matched_ratio>=.99 AND max_abs<=.01 (rtol=2^-10, atol=2^-16). The maximum-absolute condition covers every coordinate even though the ratio permits some mismatches. Saved original CPU/native coordinate[19,1986] values are -2510770.75 and -2510771.5, absolute gap .75. At either magnitude the adjacent FP32 spacing is .25; any different FP32 candidate value exceeds the .01 cap there. No candidate of the declared dtype can meet both caps for these exact saved references. The fixed fixture records source/input/criterion/reference digests and `replay.py` verifies the spacing and gap without running a product or replacing its oracle.

This Ascend950 native discriminator executed only one of55 original cases; its candidate, performance and clean cold reproduction were not measured. The observation does not assert a universal kernel/hardware precision ceiling or authorize a reference/tolerance change. Prior310p observations and their withdrawn modified-truth instance stay distinct. New inputs, dtype, reference state or authorized acceptance change need a new evidence chain. FP64 subtraction used to print the gap is only a diagnostic of the finite FP32 scalars.

## 适用场景

- **工程形态**：双 golden 接受域须按完整谓词证明是否相交；native-as-candidate 失败只是线索，有限坐标绝对界可。
- **验证过的平台**：310p、Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
