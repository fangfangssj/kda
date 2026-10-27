---
type: Runbook
title: "A phase_o5_mismatch whose re-measure reads tier1_pass=0, total=0 (zero, not a lower count) means the worker self-graded but never emitted the artifacts the canonical grader reads"
description: "A phase_o5_mismatch whose re-measure reads tier1_pass=0, total=0 (zero, not a lower count) means the worker self-graded but never emitted the artifacts the canonical grader reads"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-419"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-419 — OL-419: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness/port_a3 finalize grader contract — chip-independent); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5。
- 验证来源：soc=Ascend310P3 (dav-m200, CANN 9.0.0)。

**OL-419** — phase_o5_mismatch` whose re-measure reads `tier1_pass=0, total=0` (zero, not a lower count) = the worker self-graded the count but never EMITTED the `a5_capture.pt`/`cpu_truth_outputs.pt`/`native_capture.pt` the canonical grader reads.** The grader `precision_eval_port_a3_two_tier.load_and_classify` grades by reading emitted tensor files, NOT by trusting a printed count; a self-grading worker makes it fall through to 0/0. `0/0` (vs OL-238's `None`) is the tell-tale. Fix: emit the 3 artifacts + delegate the verdict to `load_and_classify`; never self-grade. Near-zero sub-detail: `native_capture.pt` must be a `native_kind=cpu_same_precision` dict (`NATIVE_CARVEOUT_WHITELIST`) or the OL-403 carve-out silently won't fire. Discrimination block vs OL-238/239/218/277 in the body. Anchor `precision_eval_port_a3_two_tier.py:644` + `:723`. Evidence: ge_glu_grad_v2 310p kw-1→kw-2, 9/9 T1. backend=ascendc.

- Former A5Ops ID: `OL-419`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=all (harness/port_a3 finalize grader contract — chip-independent); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5`
`verified_on: soc=Ascend310P3 (dav-m200, CANN 9.0.0)`
`unverified_on: soc=Ascend950PR (A5) / Ascend910_220x (A3) — the SAME precision_eval_port_a3_two_tier.load_and_classify grader serves port_a3 on those SoCs, so the principle transfers; the specific artifact-emission failure was observed only on the 310P port so far. Re-test the emission contract on A5/A3 before relying on it there.`

**Principle**: The canonical two-tier grader `precision_eval_port_a3_two_tier.load_and_classify` renders its verdict by READING three emitted tensor-artifact files — `a5_capture.pt` (our kernel's outputs), `cpu_truth_outputs.pt` (fp64 CPU truth), `native_capture.pt` (the reference re-run at the op's native dtype) — and comparing them in `cannbench_grader.compare.py`. It does NOT trust a count the worker prints. A worker that runs its own in-script comparison and *claims* a tier1 pass count, but never materializes those three files, makes the grader fall through to `tier1_pass=0, total=0`. That `0/0` — as opposed to a merely-lower count (the OL-238 stale-interpreter signature) — is the tell-tale: the grader had nothing to read. The fix is structural, not a count reduction: the runner/worker MUST emit the three artifacts and DELEGATE the verdict to `load_and_classify`; never self-grade. Sub-detail that bites near-zero ops: `native_capture.pt` must be a dict tagged `native_kind=cpu_same_precision` (a member of `NATIVE_CARVEOUT_WHITELIST`) for the near-zero carve-out (OL-403) to fire; an untagged or bare-list native blob is treated as `native_usable=False`, so a legitimately-correct near-zero kernel gets graded too strictly and can mismatch even when the tensors ARE emitted.

**Concrete anchor** (resolving citations):
- grader entry — `src/scripts/precision_eval_port_a3_two_tier.py:644` (`def load_and_classify`)
- the carve-out gate that reads the `native_kind` tag — `src/scripts/precision_eval_port_a3_two_tier.py:723` (`native_usable = isinstance(native_blob, dict) and native_kind in NATIVE_CARVEOUT_WHITELIST`)
- the three artifacts the grader opens — `a5_capture.pt`, `cpu_truth_outputs.pt`, `native_capture.pt` at workspace/archive root.

**Root-cause discrimination** (vs the sibling `phase_o5_mismatch` classes — pick the branch by what the re-measure shows):
- THIS entry — re-measure shows `tier1_pass=0, total=0` (zero): grader found no emitted tensor artifacts → worker self-graded. Fix: emit the 3 artifacts + delegate.
- OL-238 — re-measure shows `measured.pass_a=None` (runner stdout empty/unparseable): stale interpreter/env pointer. Fix: target-scoped `NPU_PYTHON_BIN`.
- OL-239 — re-measure shows "no Pass B verifier found locally": local-runner path divergence. Fix: mirror the missing runner paths.
- OL-218 — re-measure shows a real lower count on ONE pass while another is N/A: N/A-claim fragility. Fix: emit concrete counts for every reproducible pass.
- OL-277 — the verdict-JSON I/O contract one layer ABOVE this (how the derived verdict is emitted to stdout and materialized); THIS entry is the input-tensor-artifact layer one step below it.
