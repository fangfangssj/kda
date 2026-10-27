---
type: Runbook
title: "A pass-count's denominator is a runtime-selected artifact and the selection never appears in the score — record the dataset sha256 beside the counts, on the fallback path too"
description: "A pass-count's denominator is a runtime-selected artifact and the selection never appears in the score — record the dataset sha256 beside the counts, on the fallback path too"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-323"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-323 — OL-323: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all (verification reporting)。
- 验证来源：three independent instances measured the same day, 2026-08-01。

**OL-323** — **A pass-count's DENOMINATOR is a runtime-selected artifact and the selection never appears in the score** — `50/50` reads as a fact about the kernel when it is partly a fact about which dataset was on disk. Three carriers measured the same day: (a) `v2 canonical unavailable — FELL BACK to v1`, emitted every run, so the ratio covers v1 not canonical v2; (b) a versioned `.pt` — local 18,664,053 B → 130 cases vs on-device 22,013,086 B → 150, same source and same `model.py`; (c) ⭐ worst and easiest to miss — `verification.json` has **no dataset field at all** (`soc_version` and `phase_o5_perf_capture_version` exist), so across 8 backward archives "ran against v1" and "ran against v2" are **byte-identical** output: no warning to grep, no stale file to diff, the question is unanswerable after the fact and un-answerability reads as fine. **Rule**: record the `sha256` of the dataset/input artifact beside the counts (a version *label* is insufficient — cf. OL-299); write it on the **fallback path too**, else the new field reproduces the defect; report a count with no recorded identity as **scope-unrecorded** and never retro-label it.

- Former A5Ops ID: `OL-323`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (verification reporting)`
`verified_on: three independent instances measured the same day, 2026-08-01`

**Principle** (formulation credited to the agent who generalised it from the first two
instances): a verification score ships a numerator and a denominator, and **the denominator is
chosen at runtime from whatever artifact happens to be present.** That choice is not recorded,
so `50/50` reads as a fact about the kernel when it is partly a fact about *which dataset was on
disk when the run started*. Nothing in the score distinguishes the two.

**Three instances, three different carriers, same shape:**

| carrier | what selected the denominator | what the score showed |
|---|---|---|
| benchmark dataset version | `v2 canonical unavailable — FELL BACK to v1`, emitted every run | `50/50` — a coverage ratio over v1, not over the canonical v2 case set |
| a versioned `.pt` asset | local copy 18,664,053 B → 130 cases; on-device copy 22,013,086 B → 150 cases. Same source, same `model.py` | the local copy was treated as authoritative and a **complete round of case-count analysis was run on it and reported outward**, while the copy actually executing was the on-device one |
| archived `verification.json` | *no field at all* — `soc_version` and `phase_o5_perf_capture_version` exist, dataset identity does not | across 8 backward archives, "ran against v1" and "ran against v2" produce **byte-identical** output |

| the artifact itself | **deleted by its own author**, to free a workspace path for an A/B run | the published numbers survive in chat; nothing on disk supports them |

⇒ **Rung 2's harm is not "the denominator is unclear".** The stale artifact was believed, reasoned
from end-to-end, and the conclusion published — a wrong denominator does not announce itself as
missing information, it silently produces a complete and confident analysis.

⭐ **The rungs are ordered by how little there is left to notice.** (1) has a warning to grep.
(2) has a stale file to diff. (3) presents as *nothing at all* — no warning, no file — so the
question is not merely unanswered but **unanswerable after the fact**, and un-answerability
reads as "fine". (4) is worse again: the number still circulates in chat while the artifact it
points at is gone, so **a score that lives only in a chat log is typographically identical to
one that is backed by evidence**.

⚠️ **In (4) the motive was entirely proper** — the author deleted the workspace to reuse the
path for a controlled A/B, and reported the loss themselves. No step of it resembles destroying
evidence, which is exactly why no reviewer would have flagged it. ⇒ The countermeasure is
therefore procedural, not ethical: **copy `verification.json` + `PROGRESS` + the manifest out of
the workspace before reusing its path**, because `workspace/` is designed to be discarded
(DEBT-313) and the discard is normally correct.

**Rule.**
1. A score must carry the **identity** of what it was scored against — `sha256` of the dataset
   or input artifact, beside the counts. A *version label* is not sufficient: a label can be
   stale or simply wrong while the file is what it is (cf. OL-299, where a package's
   self-reported version SHA was a different repository's HEAD).
2. **Write it on the fallback path too, not only the success path** — otherwise the new field
   reproduces the defect it was added to fix, since the runs that most need identifying are
   exactly the ones that took the unexpected branch.
3. A count without a recorded denominator identity is reported as **scope-unrecorded**. Do NOT
   retro-label it: absent evidence for v1 and absent evidence for v2 is not evidence for
   either, and guessing converts an honest gap into a false record.

**Do not "fix" existing archives by inference.** The run that produced them is gone; the only
accurate state for them is scope-unrecorded.

**Cross-ref**: OL-299 (a self-reported version identifying the wrong object — the label-vs-file
half of rule 1), OL-311 (a negative result scoped to its probe), OL-305 (a control that measured
least passes for free), OL-321 (a reassuring result must prove its scope covers the claim — this
is that rule applied to the denominator). backend=all.
---