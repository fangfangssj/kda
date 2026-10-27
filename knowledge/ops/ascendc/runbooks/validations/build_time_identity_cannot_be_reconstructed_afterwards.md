---
type: Runbook
title: "Build-time identity cannot be reconstructed afterwards — search all refs and read the bytes as they existed in each commit before calling an archive recoverable"
description: "Build-time identity cannot be reconstructed afterwards — search all refs and read the bytes as they existed in each commit before calling an archive recoverable"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-507"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-507 — OL-507: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (archive repair and reference promotion)。
- 验证来源：soc=Ascend950PR; archive corpus + harness gates, 2026-08-18。

**OL-507** — Build-time identity cannot be reconstructed afterwards — search ALL refs and read the bytes as they existed in each commit before calling an archive recoverable. 'Absent from every commit on every ref' is the only basis for TERMINAL; recovery is then a real re-run, never a file edit or a reconstruction from a sibling/tracker. Does NOT assert generation scope: a tracker-column filter decides what the next generation produces; the reference matrix decides which existing archives may serve as a bounded reference. Read the scope list from its authority at time of use — its criterion and membership were both restated within hours on 2026-08-18, so no count is recorded here.

- Former A5Ops ID: `OL-507`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (archive repair and reference promotion)`
`verified_on: soc=Ascend950PR; archive corpus + harness gates, 2026-08-18`

Whether an archive can be repaired is decided by what was **committed at the time**, not by what a tracker
claims. If the SoC, the receipt, or the archive itself was never written to any ref, no later step can supply it
without inventing it — and an invented field is indistinguishable from a recorded one.

**Rule.** Record target/build identity at build time. Before declaring an archive repairable, search **all
refs** for the exact artifact path and read the bytes as they existed in each commit; "absent from every commit
on every ref" is the only basis for `TERMINAL`. Recovery is then a real re-run, never a file edit.

- **Scope boundary**: this entry governs archive/harness mechanics only; it does NOT determine generation eligibility. Read the current op scope from its authoritative tracker at time of use — it is a point-in-time list and was restated twice within hours on 2026-08-18.
- **positive**: `git log --all --diff-filter=AM -- <exact path>` then `git show <commit>:<path>` and parse; state
  which refs were searched.
- **anti-pattern**: reconstructing a missing receipt from a sibling archive, a tracker row, or the directory
  name; or importing an archive whose claim was rescinded by its issuer.
- **falsifier**: locating the artifact in any ref makes the archive recoverable and the TERMINAL verdict wrong —
  which is exactly why the search must be all-refs before the verdict is issued.
- ⚠️ **The ops named below are cited as REJECTED archives and their rejection reasons — anti-pattern and
  falsifier material only. None of them is an available reference; the real provider selected 0 of 5.**
- **Evidence**: 2026-08-18 — of five assigned archives, one had `build=null` in the only commit that ever
  touched its verification, one had no `verification.json` in any ref, one had `op_type` absent in every
  generation, one existed only behind a lease-issuer-rescinded claim, and one had never existed on any ref.
- **Cross-reference**: OL-503 (tracker vs eligibility), OL-505 (content binding).
