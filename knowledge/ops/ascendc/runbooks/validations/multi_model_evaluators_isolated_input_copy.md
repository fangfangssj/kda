---
type: Runbook
title: "Multi-model evaluators must give each model invocation an isolated input copy when the Model mutates its inputs in place"
description: "Multi-model evaluators must give each model invocation an isolated input copy when the Model mutates its inputs in place"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-554"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-554 — OL-554: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (host-side Python harness concern, silicon-independent by construction); cann=all; bisheng=n/a; op_class=any Model whose forward mutates its inputs (index_put_ / scatter_ / index_fill_ / masked_fill_ / fill_ / add_ family)。
- 验证来源：19_IndexPut 310p (2026-08-14) — first in-place Model on the npukernelbench-310p project。

**OL-554** — **multi-model evaluators must give each model invocation an isolated input copy when the Model mutates its inputs in place** — a shared input set feeds later models the previous model's mutated tensors (index_put_ accumulate=True → candidate graded on x+2Σ, guaranteed FAIL for ANY kernel). Signature: candidate fails ~100% of in-place cases while manual single-model runs pass → check the harness first (OL-85 logic-first), not the kernel. Fix at the harness: per-model `copy.deepcopy(raw)` + separate vendor/candidate handles (`src/scripts/precision_eval_two_tier.py:292-294`). soc=all (host-side Python, silicon-independent). 19_IndexPut 310p 2026-08-14 (first in-place Model on 310p project): post-fix 46/46 + OVERALL_T2, Pass B 123/123 inclusive. backend=all.

- Former A5Ops ID: `OL-554`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side Python harness concern, silicon-independent by construction); cann=all; bisheng=n/a; op_class=any Model whose forward mutates its inputs (index_put_ / scatter_ / index_fill_ / masked_fill_ / fill_ / add_ family)`
`verified_on: 19_IndexPut 310p (2026-08-14) — first in-place Model on the npukernelbench-310p project`

**Trigger classifier**: an evaluator invokes the SAME case through multiple models (CPU truth, vendor NPU reference, candidate kernel). If `Model.forward` mutates its inputs in place, any input tensor shared between invocations carries the previous model's writes into the next one — the candidate is graded against a state the reference never saw.

**Failure signature**: the candidate fails ~100% of in-place cases regardless of kernel quality; failure magnitude looks like the op applied one extra time (index_put_ accumulate=True → x+2Σ) or a mix of vendor and candidate writes; manual single-model runs of the same kernel pass. A candidate that fails EVERY case while independent probes show it bit-exact is an evaluator-aliasing smell, not a kernel bug — check the harness before touching kernel code (OL-85 logic-first).

**Fix at the harness, not per-op** (per the fix-harness-for-next-customer rule): give each model invocation its own deep copy of the raw inputs — one CPU tensor copy per model per case, negligible vs kernel time — and keep separate handles for vendor and candidate NPU inputs instead of one shared list.

Concrete anchor — `src/scripts/precision_eval_two_tier.py:292-294`:
```python
cpu_in  = to_cpu(copy.deepcopy(raw))
cann_in = to_npu(copy.deepcopy(raw)) if HAS_NPU else to_cpu(copy.deepcopy(raw))
ours_in = to_npu(copy.deepcopy(raw)) if HAS_NPU else to_cpu(copy.deepcopy(raw))
```

**Evidence**:
- 19_IndexPut 310p (2026-08-14): one shared `npu_in` list between `npu_model` and `cand_model` fed the candidate the CANN-mutated x — accumulate=True produced x+2Σ, a guaranteed FAIL for ANY candidate. Post-fix: Pass A 46/46 (OVERALL_T2), Pass B 123/123 inclusive. Fix landed in `src/scripts/precision_eval_two_tier.py` the same session.

**Other instances (predicted)**:
- Every future in-place benchmark op on every project/backend — any evaluator running >1 model over shared case inputs. Also applies to det-check and perf harnesses that reuse one input set across reference-then-candidate runs (a mutated input silently changes what the candidate is timed/checked on).

**Cross-reference**: OL-85 (logic-first diagnosis — this class of bug is found by diffing harness behavior, not kernel arithmetic), OL-160 (safety-net name coupling — the canonical entry-point contract this evaluator grades).

---