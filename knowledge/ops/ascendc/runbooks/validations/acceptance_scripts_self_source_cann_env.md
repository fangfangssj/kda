---
type: Runbook
title: "Every acceptance-script family member must self-source its CANN environment"
description: "Every acceptance-script family member must self-source its CANN environment"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-559"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-559 — OL-559: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (acceptance-harness / verification-script design); cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend310P3; cann=9.1.0 (sort 310p benchmark-acceptance lane, 2026-08-18)。

**OL-559** — **every acceptance-script family member must self-source its CANN environment** — sibling-script env asymmetry (precision leg self-sources set_env.sh, perf leg doesn't) produces a "precision works, perf crashes" split-brain: bare orchestrator/agent shells carry NO CANN env, and `command -v msprof` on PATH ≠ CANN PYTHONPATH → every profiled app dies in GE init (`No module named 'tbe'` → error 500001). Rule: each script self-sources set_env.sh with the ASCEND_HOME_PATH → ASCEND_TOOLKIT_HOME → latest fallback chain. sort 310p 2026-08-18 (kw-2 lost 10/10 cases × ref+asc; fixed in cannbot-skills 9bf6922d → 10/10 measured, geomean 0.814×). Cross-ref [OL-557](../performance/profiler_launched_benchmark_binds_npu_at_init.md) (device-steer sibling), OL-180. backend=all.

- Former A5Ops ID: `OL-559`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=all (acceptance-harness / verification-script design); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.1.0 (sort 310p benchmark-acceptance lane, 2026-08-18)`

**Principle**: an acceptance harness is a FAMILY of sibling scripts (precision leg,
perf leg, profiling wrappers) invoked from bare orchestrator/agent shells that carry
NO CANN environment (nothing in /etc/profile.d or the shell rc files — the CANN env
comes from the launcher process only, and a `command -v msprof` check finds msprof on
PATH without implying CANN PYTHONPATH). EVERY member of the family must self-source
CANN `set_env.sh` itself, with a fallback chain (ASCEND_HOME_PATH →
ASCEND_TOOLKIT_HOME → /usr/local/Ascend/ascend-toolkit/latest); never rely on the
caller's env. Asymmetry — one sibling self-heals, another doesn't — produces a
split-brain signature: the precision leg is green while the perf leg crashes EVERY
profiled app in GE init (`ModuleNotFoundError: No module named 'tbe'` →
`AclSetCompileopt(ACL_PRECISION_MODE)` error 500001). The failure reads like a
perf-infra mystery; it is an env-contract bug in one family member.

**Concrete anchor**: cannbot-skills 仓（公开）/ops/ops-profiling/scripts/msprof_profile_run.sh
(the perf leg that did NOT self-source — fixed to source set_env.sh with the fallback
chain, cannbot-skills commit 9bf6922d) vs its precision-leg sibling
cannbot-skills 仓（公开）/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh
(which already self-sourced).

**Evidence**: sort 310p (2026-08-18, first instance, kw-2/kw-3): the perf leg lost
10/10 cases × ref+asc to the missing self-source while the precision leg passed the
same workspace; post-fix 10/10 cases measured both sides (geomean 0.814×).

**Other instances (predicted)**: any harness script family whose members are launched
from minimally-configured shells (CI runners, agent spawns, cron) — profiling
wrappers, report generators, finalize-gate re-measure scripts.

**Cross-ref**: OL-557 (the OTHER perf-leg env pitfall from the same op's kw-3 — device
steer via `ASCEND_RT_VISIBLE_DEVICES` vs here the CANN env itself; a perf leg needs
BOTH), OL-180 (the in-process half — .so loading needs CANN env / torch_npu import
order). backend=all.
