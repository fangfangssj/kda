---
type: Runbook
title: "Literal dbL0 depth fields silently override the vendor's budget-derived double buffering"
description: "Literal dbL0 depth fields silently override the vendor's budget-derived double buffering"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-485"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-485 — OL-485: Literal dbL0 depth fields silently override the vendor's budget-derived double buffering

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul (MatmulImpl LAYER-1 kernel with hand-written TCubeTiling fields)。
- 验证来源：1_BatchMatmul db0 arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, single-variable arm, same-session consumer A/B)。

**OL-485** — measured on 1_BatchMatmul (950DT_9582): replacing literal depth-1 `dbL0A/B/C` tiling fields with the vendor's budget-derived depth formula was a measurable win. A hand literal is an override, not a neutral restatement.

- Former A5Ops ID: `OL-485`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul (MatmulImpl LAYER-1 kernel with hand-written TCubeTiling fields)`
`verified_on: 1_BatchMatmul db0 arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, single-variable arm, same-session consumer A/B)`

**Measured (the load-bearing facts, each bound to its in-repo receipt)**: exact before/after — BEFORE `t.dbL0A = 1; t.dbL0B = 1; t.dbL0C = 1;` (frozen candidate `ef528ef8`: the member named `kernel/batchmatmul_kernel.h` (member lines 119-121) inside `…/candidate_history/snapshots/f0e4be79c380a8abd1fb565d43056e076ae721d74f418d797f165a69b9df482b/ef528ef8-frozen-20260906/candidate.tar.gz`, tar blob `eda523474934`) — AFTER: sentinel `-1` at the SetTiling site (member lines 122-124) plus budget-derived depth `tiling_.dbL0A = (2 * tiling_.baseM * tiling_.baseK * sizeof(T) <= L0A_B) ? 2 : 1;` (member lines 189-191; B/C analogues) inside `…/candidate_history/snapshots/cfa2439b46e7b1b1134fbdaf00de114bc6eed3aa4dc4f40234e2d1a557cd3637/db0-20260907/candidate.tar.gz`, tar blob `f81785603548`). profiler A/B: pre-arm baseline GM 0.861716 (`performance_frozen_fleet_eval.json` blob `7ebc24875dea`) → db0 arm GM 0.870640 (`performance_db0_fleet_eval.json` blob `f3e040243194`); P51 51/51 + bit-exact + known-bad + determinism intact (`fixed51_grade.db0.json` blob `3b40271e9a0c`). Only the dbL0 fields changed between the arms.

**Mechanism (referenced, not inferred)**: 内部路径（不公开）tiling.h:458 documents `dbL0A` as "whether A matrix MTE1 is using double buffer" (default -1), and 内部路径（不公开）constant_tiling.h:127 shows the vendor computing `2*baseM*baseK*bits <= TOTAL_L0A_SIZE ? 2 : 1` — the after-state replicates that vendor formula with on-stack budget constants (L0A/B=64K, L0C=128K on 9582). Anything beyond these two cited lines (vendor behaviour on other shapes/paths) is NOT claimed.

**Evidence (repo-resolvable)**: BEFORE tar `…/snapshots/f0e4be79…/ef528ef8-frozen-20260906/candidate.tar.gz` (blob `eda523474934`), AFTER tar `…/snapshots/cfa2439b…/db0-20260907/candidate.tar.gz` (blob `f81785603548`) + `…/db0-20260907/manifest.json`; delivery merge !872 (`cb8f01c05`); baseline `…/evidence/performance_frozen_fleet_eval.json` (blob `7ebc24875dea`) and arm receipt `…/evidence/performance_db0_fleet_eval.json` (blob `f3e040243194`); grade `…/evidence/fixed51_grade.db0.json` (blob `3b40271e9a0c`) — all on main.

**Evidence (additional instance — 3-1 S7 A3→A5, 2026-09-13)**: the same dbL0 sentinel (`-1` at the SetTiling site) + on-stack budget-derived depths (`2*baseM*baseK*sizeof(T) <= L0A_B ? 2 : 1`, with 9582 budgets L0A/B=64K, L0C=128K) applied in the 3-1 A3→A5 incumbent (DSO `7f2026a4`, Ascend950DT_9582 / CANN 9.1.0); canonical P51 51/51 and 2-leg F GM 1.252 (QUALIFIED_AT_GATE). Evidence anchor in the 3-1 S7 archive (`s7_run/KB_RETRIEVAL_RECEIPT.json` OL-485 application_sites; external to this repo). Same-technique confirming datapoint, not a new rule.

**Other instances (predicted)**: kernels with hand-written `TCubeTiling` literal blocks — diff each literal field against the vendor `constant_tiling` computation for the same shape before tuning anything else.
