---
type: Runbook
title: "A fp32 \"precision FAIL\" against a fp64 golden on a multi-term-reduction op is a TRIAGE problem before it is a kernel bug — bisect the cancellation floor first, then (on vendor-suite lanes) declare the op-author tolerance from the measured envelopes"
description: "含 cancellation 元素的 fp32 算子，fp64 golden 默认阈值可能对任何正确实现都不可达（torch eager fp32 自己都超标）——先证阈值可达性再修 kernel"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-681"]
---

# A fp32 "precision FAIL" against a fp64 golden on a multi-term-reduction op is a TRIAGE problem before it is a kernel bug — bisect the cancellation floor first, then (on vendor-suite lanes) declare the op-author tolerance from the measured envelopes

`applies_to: soc=all (the bisect is numerical, not silicon); cann=9.0.0 for the ttk `Spec.tolerance` mechanism (ops-math 商发 2.1 suite); bisheng=n/a; op_class=multi_term_reduction_fp32 (softmax-attention, norm, scan — any op whose outputs include cancellation elements)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (vision_attention 310p ops-math suite, 2026-08-28 — ST 134/268 fp32 FAIL → 268/268 with the kernel UNCHANGED)`
- **Principle**: on ops whose fp32 outputs include cancellation elements (softmax-attention probabilities, multi-term accumulations), a dtype-DEFAULT fp32 grading threshold (ttk default: `stat_rel_err` 2^-13, mare limit ~1.22e-3 vs fp64 golden) is a single-rounding-class bound and can be UNREACHABLE by ANY correct fp32 implementation — on the surfacing op, torch eager fp32 itself missed the default by up to 1.3e-1 mare, and the exact-exp + fp32-storage information floor alone was 1.6e-3. The "floor, not bug" signature: **mere stays ~1e-6 (the systematic guard stays tight) while mare spreads huge**, and the max-rel-err element sits at |golden| ~ the grader's denominator floor (1e-7) with |actual − golden| ~ 1e-9 — that is the fp32 information floor, not a defect.
- **Bisect recipe (run BEFORE touching the kernel)**: (1) repro the fp64 pipeline in numpy against the golden binary — validates semantics + golden provenance; (2) simulate exact-transcendental + fp32-storage → the information-floor mare; (3) run torch eager at the INPUT dtype through the SAME grader → the competitor envelope; (4) locate the max-rel-err element and check it against the denominator floor. A real systematic defect lifts MERE; if (2) ≈ device measurement and (3) also fails the default threshold, the kernel is at the floor and no kernel edit can pass the default grade.
- **Vendor-suite fix (ttk, ops-math 2.1)**: the op-author escape hatch is a `tests/test_spec_<op>.py` (rglob'd under the ST `--plugin <op>/tests` path) declaring `tolerance = {"float32": {"standard": "stat_rel_err", "threshold": X}}` + `__spec__ = {"<op_interface_name>": "<Cls>"}`; `resolve.py` honors `Spec.tolerance[dtype].threshold` over the dtype-default table. Vendor precedent for reduction-class fp32 thresholds: layer_norm (0.004), batch_norm (0.004-0.005). `cross_check` is NOT an offline substitute (needs XPU/remote third_party; GOLDEN_FAILURE without it). **Threshold-sizing rule (this is NOT blanket tolerance-loosening)**: X must cover BOTH the kernel's worst row AND the fp32 reference's own envelope — a threshold wider than the competitor envelope is over-loosening — and mere stays on as the tight systematic guard.
- **Concrete anchor**: `代码片段 vision_attention__opsmath/test_spec_vision_attention.py.snippet.py（见文末）` (the tolerance dict + `__spec__` line; the loader/resolve path is documented in the module docstring at `:12-14`).
- **Evidence**: vision_attention 310p kw-2 (2026-08-28): the initial 134/268 ST fp32 FAIL (mare 1.467e-3 vs the 1.2207e-3 default, ~20% marginal) was root-caused to the MISSING op-author tolerance declaration, not the kernel — adding the spec moved ST to 268/268 with the kernel unchanged (`vision_attention__opsmath/verification.json`: st.rows=268, passed=268). Companion finding: on dav-m200 the hw `Exp` sat BELOW the fp32 floor in this pipeline — see the OL-103 evidence row of the same date (the Tier-1-rewrite temptation was correctly refused after the sim).
- **Other instances (predicted)**: any fp32-graded multi-term-reduction op against a fp64-golden suite (softmax/attention variants, norm-family, long accumulations) on ANY SoC — the bisect ordering (floor sim + competitor envelope BEFORE kernel rewrite) is arch-independent; only the tolerance-declaration mechanism is ttk-specific.
- **Cross-ref**: [`OL-103`](ol_103_npu_transcendentals_fp16_precision.md) (per-SoC transcendental grade — sim before Tier-1 rewrite), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) / OL-627 (no tolerance loosening without evidence — this entry's sizing rule IS the evidence form), [`OL-664`](../performance/ol_664_st_green_is_precision_coverage_only_the_ttk_manu.md) (the ttk ST lane's other blind spots), OL-321 (prove the instrument's scope before trusting its red). backend=all (bisect principle) / ascendc ops-math-suite (the `Spec.tolerance` mechanism).

## 内嵌代码片段

### vision_attention__opsmath/test_spec_vision_attention.py.snippet.py

```python
  After the 2026-08-31 cube-path rewrite (CubeQK/PV), bf16 rows fail the
  dtype-default stat_rel_err (2^-7; mare limit 10x = 0.078) with
  mere ~2e-3 PASS but mare ~49.6 FAIL (va310p_bf16_scale_explicit,
  deterministic, bit-identical across runs). The mare driver element is a
  catastrophic-cancellation output: |g| ~ 2e-7, |a-g| ~ 1.6e-5 — the
  relative-error metric explodes on near-zero goldens while the absolute
  deviation is 3 orders below any acceptance atol.

  Floor proof that the default is UNREACHABLE by any bf16 implementation
  (the bf16 twin of the fp32 floor proof above): perfect-fp64 softmax with
  the MANDATORY bf16 P-matrix storage (cube PV path requires bf16 operands)
  + fp32 accumulate + bf16 output storage, on the on-disk ST bins of
  va310p_bf16_scale_explicit, gives mere 1.4e-2 / mare 407.9 vs the golden
  (golden = fp64 math rounded to bf16 exactly; pure-fp64+bf16-out-only
```
