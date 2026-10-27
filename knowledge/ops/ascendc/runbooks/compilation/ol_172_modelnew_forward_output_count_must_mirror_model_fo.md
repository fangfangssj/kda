---
type: Runbook
title: "ModelNew.forward output count MUST mirror Model.forward output count — pybind aux outputs are discarded at the ModelNew wrapper boundary [V351+V220, ALL_MODES, safety-net + reference-contract-parity]"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=any_op_with_aux_outputs_at_kernel_pybind_boundary (e.g. attention forward returning attn_out+softmax_max+softmax_sum, topk returning values+indices"
tags: [task.debugging, topic.precision, topic.implementation.pipeline, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-172-modelnew-forward-output-count-must-mirror-model-fo.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-172"]
---

# ModelNew.forward output count MUST mirror Model.forward output count — pybind aux outputs are discarded at the ModelNew wrapper boundary [V351+V220, ALL_MODES, safety-net + reference-contract-parity]

> 原 OKF v1 卡号：**OL-172**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=any_op_with_aux_outputs_at_kernel_pybind_boundary (e.g. attention forward returning attn_out+softmax_max+softmax_sum, topk returning values+indices, scatter returning output+aux)`
`verified_on: a5_ops:3_FusionAttention kw-2 spawn-1 2026-05-21 — canonical Pass A evaluator surfaced "output count mismatch: cpu=1 cann=1 ours=N" and failed ALL 61 cases when ModelNew.forward returned the kernel's 7-tuple instead of the single attention_out tensor`

**Principle**: The canonical verification path treats `Model.forward` (workspace reference) as the ground-truth output shape contract. `ModelNew.forward` (our kernel's nn.Module wrapper) MUST emit EXACTLY the same number of output tensors in the same order. If the underlying AscendC kernel's pybind layer returns auxiliary tensors (e.g. softmax statistics for FA, sort indices, debugging counters), those auxiliary outputs MUST be discarded inside the `ModelNew.forward` body — they do not propagate to the verifier.

This is distinct from (and orthogonal to) OL-160's file-name-parity rule. OL-160 governs WHICH files exist (`model.py` + `model_new_ascendc.py`); OL-172 governs the OUTPUT INTERFACE inside `ModelNew.forward`. Both must hold simultaneously.

**Symptom (caught by canonical evaluator)**:
```
output count mismatch: cpu=1 cann=1 ours=7
```
All cases fail — the comparator can't iterate when output counts differ. This is NOT a precision failure; the kernel may be numerically correct and still report 0/N PASS_T1 purely because the wrapper's output shape doesn't match the reference contract.

**Concrete anchor** (correct ModelNew.forward when kernel pybind returns aux outputs):
```python
class ModelNew(nn.Module):
    def forward(self, q, k, v, ...):
        # Internal kernel returns 7-tuple matching vendor multi-output contract:
        attn_out, softmax_max, softmax_sum, _, _, _, _ = _ext.run_fusion_attention(q, k, v, ...)
        # But workspace/model.py emits a SINGLE tensor — wrapper must mirror that contract:
        return attn_out  # discard aux; do NOT return the tuple
```

**Anti-pattern (BANNED)** — propagating kernel pybind's full tuple under the rationale "match the vendor API":
```python
class ModelNew(nn.Module):
    def forward(self, q, k, v, ...):
        return _ext.run_fusion_attention(q, k, v, ...)  # 7-tuple → all cases FAIL
```
Rationale being false: the canonical evaluator never compares against the vendor API directly — it compares against `Model.forward`. The vendor API's tuple-shape is irrelevant; what matters is wrapper-vs-reference parity.

**Alternative resolution path** (when the aux outputs genuinely matter for downstream consumers):
- Update `workspace/model.py` so `Model.forward` ALSO returns the same multi-output contract. This makes the contract uniform top-to-bottom and aux outputs propagate to the verifier (which then validates them too).
- Choose this path only when the downstream user genuinely needs the aux tensors AND the workspace reference algorithm can naturally emit them — otherwise the discard-at-wrapper form is simpler.

**Detection** (post-archive scan, similar to OL-160 file-name gate):
```python
# In scan_delegation_cheating.py or finalize_pipeline gate:
import ast, inspect
ref_outputs = count_return_tensors(workspace_path / "model.py", "Model", "forward")
new_outputs = count_return_tensors(workspace_path / "model_new_ascendc.py", "ModelNew", "forward")
assert ref_outputs == new_outputs, f"OL-172 violation: Model.forward returns {ref_outputs}, ModelNew.forward returns {new_outputs}"
```
Counting returns statically requires tracking `return` expression shape (tuple vs single name vs subscript) — straightforward AST walk.

**Cross-ref**: OL-160 (canonical entry-point file names — sibling rule about WHICH files exist), OL-165 (forward() must exercise the kernel — sibling rule about WHAT forward() does); CLAUDE.md "No PyTorch/CANN Delegation"; canonical-evaluator comparator semantics (`precision_eval_two_tier.py`).

**Other instances (predicted)**: any fused op whose underlying kernel emits per-stage debugging/statistics tensors (FA softmax stats, scaled dot-product softmax denom, topk indices alongside values, scatter aux). Also applies whenever the kernel team intentionally returns more outputs than the user-facing API contract — for diagnostics, recomputation skipping, or backward-pass reuse — and a wrapper boundary collapses them down to user-visible outputs.

**Evidence**: 3_FusionAttention kw-2 spawn-1 2026-05-21: kernel pybind returned 7-tuple matching `torch_npu.npu_fusion_attention` vendor contract; ModelNew.forward also propagated the tuple → 0/61 PASS_T1 across all cases via `output count mismatch: cpu=1 cann=1 ours=7`. Fix: ModelNew.forward returns only the first element. After fix, case_3 reaches PASS_T1 with ours_mere=1.999e-6 < cann_mere=2.227e-6.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-172（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
