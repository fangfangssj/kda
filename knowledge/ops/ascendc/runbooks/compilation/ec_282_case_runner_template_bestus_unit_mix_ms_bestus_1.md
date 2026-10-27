---
type: Runbook
title: "case-runner template `bestUs` unit-mix — `ms < bestUs*1000.0` compares ms against us (almost always true) → the timing field reports the LAST repeat, not the min [ops-math case-pack runner template — FIXED in template]"
description: "case-pack runner 的 `us` 计时字段取的是最后一次 repeat 而非最优值——harness 级 bug，已修；读该字段做性能判断会错。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-282", "bestUs"]
---

# case-runner template `bestUs` unit-mix — `ms < bestUs*1000.0` compares ms against us (almost always true) → the timing field reports the LAST repeat, not the min [ops-math case-pack runner template — FIXED in template]

`applies_to: soc=all; cann=9.0.0 ops-math 2.1 case-pack runner; bisheng=n/a; op_class=all ops shipped through the case-pack runner`
`verified_on: soc=Ascend310P3; cann=9.0.0 (conv2d_backprop_input 310p, 2026-08-26)`
- **Symptom**: case-pack runner `us` timing field equals the last repeat's time, not the best-of-repeats; harmless for determinism (the FNV hash is per-repeat) but any perf reading off the field is wrong.
- **Root cause**: `bestUs = ms < bestUs * 1000.0 ? ms * 1000.0 : bestUs;` — `ms` (milliseconds) compared against `bestUs * 1000.0` (microseconds); with `bestUs` seeded at 1e30 the branch is almost always taken, so `bestUs` tracks the last repeat.
- **Fix**: harness-level, landed same day (per the fix-the-harness rule, not documented-and-left): `ops_math_case_runner_template.cpp:289`（a5_ops 路径，未迁移） now converts once (`const double us = ms * 1000.0; if (us < bestUs) bestUs = us;`). Workspaces holding a pre-fix generated runner should regenerate from the template.
- **Evidence**: conv2d_backprop_input 310p (2026-08-26, kw-1 respawn perf investigation): observed last-repeat timing in `conv2d_backprop_input__opsmath/evidence/case_runner_run.log`; template fixed 2026-08-26.
- **Cross-ref**: OL-371 (the ST/case-runner lane's perf blind spots — a wrong timing field compounds them), (the measurement-trap family). backend=all (harness template).
