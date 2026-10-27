---
type: Runbook
title: "A gate with zero production callers is inert, and its own tests cannot tell you so — helper-direct tests pass against dead code"
description: "A gate with zero production callers is inert, and its own tests cannot tell you so — helper-direct tests pass against dead code"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-506"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-506 — OL-506: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (finalize hooks, provider checks, admission gates, recovery routines)。
- 验证来源：soc=Ascend950PR; archive corpus + harness gates, 2026-08-18。

**OL-506** — A gate with zero production callers is inert, and its own tests cannot tell you so — helper-direct tests pass against dead code. Assert through the production entry point, count callers mechanically, and separate discriminators (red when unwired) from guards (green in both worlds). Anchor: a contract shipped with 25 green tests and 0 callers, found by one `git grep ':!tests'`. Does NOT assert generation scope: a tracker-column filter decides what the next generation produces; the reference matrix decides which existing archives may serve as a bounded reference. Read the scope list from its authority at time of use — its criterion and membership were both restated within hours on 2026-08-18, so no count is recorded here.

- Former A5Ops ID: `OL-506`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (finalize hooks, provider checks, admission gates, recovery routines)`
`verified_on: soc=Ascend950PR; archive corpus + harness gates, 2026-08-18`

A check that nothing calls still passes every test written against it, because those tests call it directly.
Green suites, a documented rule, and a reviewer reading the diff all fail to distinguish "wired" from
"unreachable". The only thing that separates them is entering through the real caller.

**Rule.** For any new gate, assert through its **production entry point**, and check the caller count
mechanically. Keep the entry arms honest by separating discriminators (red when unwired) from guards (green in
both worlds, asserting an absence of harm) — a guard counted as a discriminator inflates the evidence.

- **Scope boundary**: this entry governs archive/harness mechanics only; it does NOT determine generation eligibility. Read the current op scope from its authoritative tracker at time of use — it is a point-in-time list and was restated twice within hours on 2026-08-18.
- **positive**: `git grep -l <module> src/scripts ':!*tests*'` returns the intended callers; entry-point arms
  are red against the unwired commit and green after wiring.
- **anti-pattern**: a helper-direct test suite offered as evidence of integration; or a docstring/PR body
  asserting "two callers" that the code does not have.
- **falsifier**: an entry-point arm that stays green after its production caller is deleted is not testing the
  wiring, and must be rewritten.
- **Evidence**: 2026-08-18 — a contract shipped with 25 passing tests and **zero** production callers, found by
  one `git grep ... ':!tests'`. After wiring: 11 entry arms red at the unwired commit, 13 green after, with 2
  explicitly labelled guards. Same shape recorded independently in DEBT-483 (a gate with no production callers
  and no plugin declaring a role) and in a recovery routine that was implemented, unit-tested, and called by
  nothing.
- **Cross-reference**: DEBT-483, OL-504.
