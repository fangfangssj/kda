---
type: Runbook
title: "msprof_perf_summary.py tensor random-synth crashes on `shape: null` optional tensor inputs — null shape must emit `None` (ABSENT), not a synthesized shape [shared perf-infra, input-space gap]"
description: "msprof 脚本把 `shape: null` 的张量输入当成\"随机合)` 进 randint）——数据合同坑。"
tags: [task.debugging, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-182", "shape:", "None"]
---

# msprof_perf_summary.py tensor random-synth crashes on `shape: null` optional tensor inputs — null shape must emit `None` (ABSENT), not a synthesized shape [shared perf-infra, input-space gap]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; msprof_perf_summary.py (shared perf-infra serializer); op_class=any benchmark op whose case json carries an OPTIONAL tensor input as a tensor-typed entry with "shape": null and no explicit "value" (observed: 11_DequantSwiGluQuant — absent `bias` int32 + `quant_offset` fp32)`

- **Severity**: HIGH for canonical perf of the affected op class — the generated wrapper dies at runtime on the first null-shape-optional case (`torch.randint(-100, 100, None, ...)` TypeError), so no canonical perf artifacts are produced for the whole run. Independent of, and stackable with, PB-179 (the scalar-attr dtype-tag gap in the same serializer).
- **Symptom**: `_serialize_jsonl_inputs` tensor branch (cannbot-skills 仓（公开）/ops/ops-profiling/scripts/msprof_perf_summary.py:810-819) treats a tensor-typed input whose `shape` is `null` as a shape to random-synthesize: `shape_expr = str(None)` → the wrapper executes `torch.randint(-100, 100, None, dtype=...)` → `TypeError: randint(): argument 'size' ... NoneType`.
- **Mechanism**: benchmark_acceptance's null-shape convention is "`shape: null` == ABSENT optional input" (`_detect_broadcast` null-shape=absent rule), but the msprof JSONL random-synth path does not apply that rule to tensor-typed entries — only the attr/scalar branch honors an explicit `"value": null` → emits `None`. A tensor-typed absent optional (no value, null shape) falls through to shape synthesis. Note `_case_has_empty_tensor` already skips `shape is None` (does not classify it as an empty tensor) — the synth path just never got the same guard.
- **Fix (generic, verified on the 310p run)**: in the tensor branch, treat `shape is None` as ABSENT and emit `None` for that input instead of synthesizing — the same contract the attr branch uses for an explicit null value. PROVENANCE GAP: the pinned `cannbot-skills` submodule copy in this repo does NOT yet carry the guard (submodule working tree clean, no committed fix); the fix was verified on the run-side copy only — re-apply / re-commit the None-emit patch to the submodule before the next op whose case json carries null-shape tensor optionals.
- **Detection**: canonical quick run's wrapper exits with the randint-None TypeError on the first null-shape-optional case while the op's own `Model.forward` handles the same case (treats the optional as absent) natively → this PB, not a kernel/model bug.
- **Evidence**: 11_DequantSwiGluQuant 310p (2026-09-02, benchmark-acceptance lane, failures_ledger infra row): crash hit + generic fix applied to the run copy; with the fix the perf run completed (asc legs captured, ref legs N/A per PB-80's Path-A structural half, perf-script rc==0). Anchor: `dequant_swiglu_quant/failures_ledger.md:10` + `dequant_swiglu_quant/knowledge_update.md:11` finding 4.
- **Cross-ref**: [`PB-179`](pb_179_msprof_perf_summary_py_jsonl_scalar_value_crashe.md) (sibling input-space gap + stray-jsonl hijack in the same serializer), PB-80 (Path-A ref N/A stacks with this). backend=ascendc.

## 适用场景

- **工程形态**：msprof 脚本把 `shape: null` 的张量输入当成"随机合)` 进 randint）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
