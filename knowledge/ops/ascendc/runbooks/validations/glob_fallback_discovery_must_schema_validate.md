---
type: Runbook
title: "Glob-fallback artifact discovery must schema-validate each candidate file"
description: "Glob-fallback artifact discovery must schema-validate each candidate file"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-558"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-558 — OL-558: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness / acceptance-script design); cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend310P3; cann=9.1.0 (reduce_sum 310p benchmark-acceptance lane, 2026-08-17)。

**OL-558** — **glob-fallback artifact discovery must schema-validate each candidate file.** A `*.jsonl` fallback over a shared workspace ingests manifests/event logs as "cases" (64 ≠ 37); index-based consumers run out of range and silently degrade to synthetic inputs. Defenses: ship the priority-named case file (`<bench>_perf_cases.jsonl`) so the fallback never fires, AND schema-validate (`inputs` key) with a loud fail on zero valid cases. reduce_sum 310p 2026-08-17 (`_load_cases_jsonl` globbed `reference_manifest.jsonl`). backend=all.

- Former A5Ops ID: `OL-558`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=all (harness / acceptance-script design); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.1.0 (reduce_sum 310p benchmark-acceptance lane, 2026-08-17)`

**Principle**: when a harness script discovers its input set by a fallback glob over a
workspace directory (`*.jsonl`, `*.json`, `*_manifest*`), the glob does not know which
files are cases and which are sibling artifacts (reference manifests, orchestrator
event logs, perf reports). Every match is then silently accepted as a case; index-based
consumers (`get_input_groups()[case_idx]`) run out of range and — worse — degrade to a
synthetic fallback instead of failing loudly. Two cheap defenses, apply BOTH: (a)
ship the priority-named case file (`<bench>_perf_cases.jsonl`-style) in the op
workspace so the fallback never fires, including in the gate re-run environment; (b)
schema-validate each globbed file (e.g. requires an `inputs` key) before accepting it,
and fail loud on zero valid cases rather than synthesizing inputs.

**Evidence**: reduce_sum 310p (2026-08-17, first instance): `_load_cases_jsonl`'s
`*.jsonl` fallback picked up `reference_manifest.jsonl` — ~64 "cases" ≠ the 37 real
benchmark cases; wrappers indexed out of range → None → synthetic fallback inputs.
Fix shipped: `<bench>_perf_cases.jsonl` travels with the workspace. Harness-level
schema validation is the remaining hardening (not yet implemented at merge time).

**Other instances (predicted)**: any workspace-local discovery glob (edge datasets,
reference captures, manifest-driven replays) — the shared-workspace layout guarantees
sibling artifacts exist.

**Cross-ref**: OL-557 (the sibling acceptance-perf-lane pitfall from the same op —
garbage measurement via env, vs here via wrong case set). backend=all.
