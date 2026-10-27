---
type: Runbook
title: "A timing domain that cannot reproduce its own reference leg cannot grade arms — bind the F metric to the profiler device-task domain"
description: "A timing domain that cannot reproduce its own reference leg cannot grade arms — bind the F metric to the profiler device-task domain"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-487"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-487 — OL-487: A timing domain that cannot reproduce its own reference leg cannot grade arms — bind the F metric to the profiler device-task domain

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul (µs-scale batched matmul); instrument=aog_perf_eval profiler V5 (kernel_details.csv sum(Duration)/active_count over AI_CORE+AI_VECTOR_CORE) vs time.perf_counter wallclock with per-call synchronize。
- 验证来源：1_BatchMatmul tilev4 window (2026-09-07, dev6) + l3_tinygate four-arm takeover window (2026-09-08, Ascend950DT host card6)。

**OL-487** — one measured divergence case: on 1_BatchMatmul at µs-per-call scale, wallclock per-call timing was noise-dominated and priced the host launch path (pybind custom kernel vs `aclnnBatchMatMul`), while the profiler device-task domain separated the arms with a self-consistent reference leg.

- Former A5Ops ID: `OL-487`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul (µs-scale batched matmul); instrument=aog_perf_eval profiler V5 (kernel_details.csv sum(Duration)/active_count over AI_CORE+AI_VECTOR_CORE) vs time.perf_counter wallclock with per-call synchronize`
`verified_on: 1_BatchMatmul tilev4 window (2026-09-07, dev6) + l3_tinygate four-arm takeover window (2026-09-08, Ascend950DT host card6)`

**Measured (the load-bearing facts)**: takeover four-arm window — wallclock geomean main 0.7715 / db0 0.7678 / L3 0.7231 / graft 0.7668 (all <1.0, no arm separation), wallclock reference-leg per-case spread 0.537–1.818; profiler device-task geomeans main 0.9530 / L3 1.1330 / graft 1.1332, profiler reference-leg agreement 0.9992. tilev4 window — the tilev4 receipt itself shows the same divergence: profiler GM 0.887975 vs wallclock GM 0.680995 (`performance_tilev4_fleet_eval.json` blob `d3b93f923b67`), and the earlier db0 receipt shows the same sign (profiler 0.870640 vs wallclock 0.861623, `performance_db0_fleet_eval.json` blob `f3e040243194`).

**Method guidance (narrow)**: on this op/instrument, when wallclock and profiler disagree, check the reference-leg spread first — if the reference cannot reproduce itself in a domain, that domain cannot grade arms. This entry does NOT claim wallclock is unusable on other ops/hosts/scales and does NOT claim 950-class gates can only use the device-task domain; it records one measured divergence and the check that resolved it.

**Evidence (repo-resolvable)**: `output/npukernelbench/src/kernels/1_BatchMatmul/l3_tinygate_20260908T2030Z/` README §4.3 + `f_arms/perf_result.*.json` (four arms, same window); tilev4-window receipts `output/npukernelbench/src/kernels/1_BatchMatmul/evidence/f51_tilev4fix.json` (blob `e051588b7dfe`) and `…/evidence/fixed51_grade.tilev4.json` — all on main.

**Other instances (predicted)**: µs-scale ops whose F could flip on host jitter — measure both domains once and keep the reference-leg spread as the domain-validity check.
