---
type: Runbook
title: "msprof_perf_summary.py `_jsonl_scalar_value` crashes on attr inputs declared `dtype:\"int\"` whose value is a LIST — canonical quick run dies at the first such case [shared perf-infra, input-space gap]"
description: "msprof_perf_summary 的 `_jsonl_scalar_value` 对 list 型标量直接 int() 崩溃——整进程死亡而非单 case 报错。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.programming_model.execution_unit.scalar, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-179", "dtype:"]
---

# msprof_perf_summary.py `_jsonl_scalar_value` crashes on attr inputs declared `dtype:"int"` whose value is a LIST — canonical quick run dies at the first such case [shared perf-infra, input-space gap]

`applies_to: msprof_perf_summary.py (shared, untouchable per owner directive); op_class=any benchmark op whose JSON declares an attr input with a list value but an int dtype tag (observed: 32_AdaptiveAvgPool3d — 6/100 cases tagged "int" with value [D,H,W], 94/100 consistently tagged "list")`
`unverified_on: whether other vendor JSONs mix the tags the same way`
- **Severity**: HIGH for canonical perf of the affected op class — the run aborts (exit 1) at the first int-tagged case, so no canonical performance/log/perf_report artifacts are produced at all. Independent of, and stacking with, PB-80 (which makes the remaining cases N/A anyway).
- **Symptom**: `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'` from `_jsonl_scalar_value` via `_serialize_jsonl_inputs` → `_generate_wrapper_script` → `_measure_one_impl_quick` → `run_quick_mode` (whole-process death, not a per-case error row).
- **Mechanism**: vendor benchmark JSON tags the same `output_size` attr inconsistently across cases — `{"dtype":"list","value":[1,2,3]}` (case 0, serializes fine via the `return val` fallback) vs `{"dtype":"int","value":[9,32,417]}` (case 1, hits `int(val)`). The serializer trusts the dtype tag for coercion and does not check `isinstance(val, list)`.
- **Workaround (metadata-only, disclosed)**: the script's own case discovery prefers `*_perf_cases.jsonl` over `*.json` (`_load_cases_jsonl` first-priority glob) — emit the benchmark cases verbatim with ONLY the attr dtype tag normalized to `"list"` where the value is a list (values/shapes/order/count untouched, verified programmatically). Run the canonical entry against that file. The run then completes (asc legs capture normally; on Path-A ops ref legs stay N/A per PB-80's structural half).
- **Detection**: canonical quick run exits 1 with the int(list) traceback while the op's own `model.get_input_groups()` handles the same case natively → this PB, not a kernel/build problem.
- **Evidence**: 32_AdaptiveAvgPool3d 310p (2026-08-21): archive-root run mis-loaded `orchestrator_events.jsonl` (a stray `.jsonl` in the archive root — see companion note) and "succeeded" with 20 garbage `?`-shape N/A rows; pointed at msprof_verify/ it read the real JSON and crashed at case 1 as above.
- **Companion note (same evidence chain)**: `_load_cases_jsonl` globs ANY `*.jsonl` in the output dir before falling back to `*.json` — an archive containing a stray `.jsonl` (e.g. an orchestrator event log) silently profiles the wrong "cases". Keep non-case `.jsonl` files out of the perf output dir.
- **Cross-ref**: PB-80 (stacks with this — same op class), DEBT-362 (ROADMAP §6). backend=ascendc.
