---
type: Runbook
title: "Check a component's claimed share against its own msprof pipe budget before quoting it — one division rejects harness artefacts and bad extrapolations"
description: "Check a component's claimed share against its own msprof pipe budget before quoting it — one division rejects harness artefacts and bad extrapolations"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-330"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-330 — OL-330: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

Check a component's claimed share against its own msprof pipe budget before quoting it — one division rejects harness artefacts and bad extrapolations

- Former A5Ops ID: `OL-330`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`

**Principle.** msprof reports per-pipe time for a kernel. A component that executes on one pipe cannot cost
more than that pipe's total, so **the profile carries a free upper bound on every attribution claim**. One
division rejects both harness artefacts and bad extrapolations, and it costs nothing.

**The measured instance.** For one stage the pipe split was `aiv_scalar 0.839 / aiv_vec 0.107 / mte2 0.048 /
mte3 0.047`. A probe reported transcendentals — vector-pipe work — at 98.5% of the stage: **9.2x the entire
vector budget**, so it was measuring its own memory-bound harness. The same check would have rejected an
extrapolation of mine that put nine GM writes at ~6% against a 4.79% MTE3 ceiling; measured, they were 0.43%.

**Decision rule.** Before reporting "component X is N% of this kernel", compute `N% x duration` and compare
it to X's own pipe column. Over budget ⇒ report the discrepancy, not the share. Under budget ⇒ the claim is
admissible and the remaining pipe headroom bounds what is still unattributed. Applied across four
independent probes, the surviving shares summed to 84.2% against a measured `aiv_scalar + aiv_vec` of 94.6%
— agreement that no single probe could have established alone.

- **Evidence**: FlashKDA A5 port, 2026-08-01, four concurrent probes on separate exclusive NPUs.
- **Other instances (predicted)**: any fan-out where several agents each report a share of one total — the
  shares need not sum to 100% (pipes overlap, components are unmeasured), so a per-pipe bound is the only
  cheap cross-check available to the caller.
