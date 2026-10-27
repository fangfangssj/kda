---
type: Runbook
title: "A PASS in a tracking spreadsheet is not archive eligibility — report historical verdict and current provider eligibility as two labelled layers"
description: "A PASS in a tracking spreadsheet is not archive eligibility — report historical verdict and current provider eligibility as two labelled layers"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-503"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-503 — OL-503: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (archive/reference admission driven from an external tracker; harness-metadata mechanism, not device behaviour)。
- 验证来源：soc=Ascend950PR; archive corpus + harness gates, 2026-08-18。

**OL-503** — A PASS in a tracking spreadsheet is not archive eligibility. Report historical verdict and current provider eligibility as two labelled layers, each with its own denominator and observation point; never transcribe a tracker PASS into a provider verdict, and never let a tracker row justify widening an admission gate. Anchor: tracker 5/5 historical double-PASS vs the same day's real provider seen=3 admitted=0 selected=0. Does NOT assert generation scope: a tracker-column filter decides what the next generation produces; the reference matrix decides which existing archives may serve as a bounded reference. Read the scope list from its authority at time of use — its criterion and membership were both restated within hours on 2026-08-18, so no count is recorded here.

- Former A5Ops ID: `OL-503`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (archive/reference admission driven from an external tracker; harness-metadata mechanism, not device behaviour)`
`verified_on: soc=Ascend950PR; archive corpus + harness gates, 2026-08-18`

A migration tracker records what a generation once achieved. An archive provider decides what may serve as a
reference **now**, from receipts committed in the repository. These answer different questions and can disagree
completely, so a report that gives one number is ambiguous about which question it answered.

**Rule.** State historical verdict and current eligibility as two labelled layers, each with its own
denominator and observation point. Never transcribe a tracker PASS into a provider verdict, and never let a
tracker row justify widening an admission gate.

- **Scope boundary**: this entry governs archive/harness mechanics only; it does NOT determine generation eligibility. Read the current op scope from its authoritative tracker at time of use — it is a point-in-time list and was restated twice within hours on 2026-08-18.
- **positive**: "Layer 1 tracker double-PASS 5/5; Layer 2 current-main provider seen=3, admitted=0,
  selected=0" — both layers, both denominators, both timestamps.
- **anti-pattern**: "five ops passed, so five references are ready." One archive was never committed to any
  ref, one exists only behind a rescinded claim, and three are present but rejected.
- **falsifier**: a provider run that reports `selected > 0` for an op whose canonical receipts are absent would
  show the layers are not independent, and this entry would be wrong.
- **Evidence**: 2026-08-18 — tracker pinned at sha256 `6a924906…`, 5/5 historical double-PASS; the real
  provider on the same day's `main` returned seen=3 admitted=0 selected=0 for the same five ops.
- ⚠️ **A third axis exists and is routinely confused with these two.** "Which ops should the
  next generation produce" and "which existing archives may serve as a bounded reference" are
  different questions with different membership, and their overlap is partial. On 2026-08-18
  the generation filter was restated twice within hours — its criterion, its membership and
  its overlap with the reference matrix all changed — so neither the counts nor the members
  are recorded here: any value written would have been false within the hour. State which
  question a list answers before using it as a filter, and read the list from its authority.
- **Cross-reference**: OL-504 (a mandate with no consumer), OL-507 (identity is not reconstructible).
