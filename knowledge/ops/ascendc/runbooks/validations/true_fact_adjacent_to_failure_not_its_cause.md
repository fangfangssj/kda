---
type: Runbook
title: "A true fact adjacent to a failure is not its cause — and it is harder to retract than a wrong premise"
description: "A true fact adjacent to a failure is not its cause — and it is harder to retract than a wrong premise"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-306"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-306 — OL-306: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (root-cause discipline)。
- 验证来源：one multi-agent failure investigation, 2026-07-31 -- three successive whole-object causes proposed and each refuted within the same session。

**OL-306** — **A true fact ADJACENT to a failure is not its cause; and it is harder to retract than a wrong premise, because re-checking the measurement confirms it while the assumed relevance is never what anyone re-checks.** One investigation, three whole-object causes, all refuted in one session (2026-07-31): "their startup script diverged" (self-diagnosed, agreed to instantly, in team protocol within 20 min → refuted by the author's own control: the reference run verbatim on the same machine failed identically) → "this machine's collective-comm environment" (asserted in the same act as retracting the first → refuted 15 min later by a successful run there) → "the rank table disagrees with this host's device plane" (carried a REAL measurement; only its relevance was assumed). **Actual cause: two missing driver-environment lines, absent from all four failed attempts.** Persuasiveness rose down the list while correctness did not. Companion: two individually-sourced premises joined across **two different functions** that shared a parameter name, with the connecting chain never read ⇒ **precedent is not evidence; a shared identifier is not a shared contract.** **Rule**: write "F causes X only if C1…Cn hold", then verify the cheapest Ci — here the unasked one-command question was "did the failing attempts even traverse the multi-device path that reads that config?". Distinguish **measured** / **measured-but-relevance-unverified** / **inferred**; the middle class has no natural opponent.

- Former A5Ops ID: `OL-306`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (root-cause discipline)`
`verified_on: one multi-agent failure investigation, 2026-07-31 -- three successive whole-object causes proposed and each refuted within the same session`
**Principle**: an investigation produces facts faster than it produces causes, and a fact
discovered *next to* a failure acquires unearned causal status. The dangerous version is not
a guess — it is a **correct measurement whose relevance is assumed**. It resists correction
precisely because re-checking the measurement confirms it, and the unverified step (that this
measured thing is what broke the run) is never the thing anyone re-checks.
**Instance — three causes for one failure, in increasing order of persuasiveness, all wrong.**
```
1. "the author's startup script diverged from the reference"  <- self-diagnosed, agreed to
   immediately, written into team protocol within 20 min, refuted by the author's own
   control: running the reference verbatim on the same machine failed identically
2. "this machine's collective-communication environment"      <- proposed in the same act
   as retracting (1); refuted 15 min later by a successful run on that machine
3. "the rank table's addresses disagree with this host's real device plane"  <- carried a
   REAL measurement (the live interface addresses genuinely differed); only its relevance
   was assumed
ACTUAL cause: two missing driver-environment lines, absent from all four failed attempts;
adding them made the run succeed immediately.
```
All three share one shape: **from a single failing run, infer a cause attached to a whole
object** ("their script", "this machine", "that config path"). Persuasiveness increased down
the list while correctness did not, and #3 was the hardest to withdraw because deleting it
felt like deleting a measurement.
**Companion failure in the same session — two sourced facts, an unsourced join.** A proposal
was built from (a) one line of source showing a value is returned, and (b) a behaviour
*verified by measurement* on a second function. Both premises were real. They described **two
different functions** that happened to share a parameter name, and the chain connecting them
was never read. ⇒ **Precedent is not evidence**, and a shared identifier is not a shared
contract. The same pattern appeared as: "every dimension is green" where no single test
covered the load-bearing dimension — each link true, the chain carrying nothing.
**Rule**: when about to cite fact F as the cause of failure X, write the sentence *"F causes
X only if C1…Cn all hold"*, then verify the cheapest Ci. In the instance above the missing
question — "did the failing attempts even traverse the multi-device path that reads that
config?" — was one command, and nobody asked it. Distinguish in the write-up between
**measured**, **measured-but-relevance-unverified**, and **inferred**; the middle category is
the one that has no natural opponent.
**Second instance — the adjacent fact was a REAL COVERAGE GAP, which is the strongest form
(2026-08-01, KB tooling).** Observation: a newly landed `OL-` entry had no `KB_INDEX` row while
its immediate predecessor did. Cited cause: the index audit only scans `target/<backend>/` and
therefore cannot see `shared/ALWAYS_LOADED_RULES.md`, so it reports *0 orphans* while an entry
sits unindexed. **That gap is real, reproducible, and was irrelevant.** Actual cause: every
other always-loaded `OL` has a *canonical* twin in `OPERATIONAL_KNOWLEDGE.md` with the short
form in `shared/` — the new entry had only the short form, so what was missing was not an index
row but the canonical entry itself.

⭐ **Why this variant outranks a guess.** A guess invites a check. A real defect supplies an
explanation that is *internally consistent, independently verifiable, and still wrong about
this failure* — and because verifying it succeeds, the verification feels like confirmation.
⇒ **A plausible explanation is cheaper to accept than a correct one, precisely because it
costs nothing to confirm.** The unasked question was again one command: *"do comparable
entries appear in the index, and by what route?"* — which would have shown the canonical/short
pair and located the real gap immediately.

⇒ **Two aggravating details, both about how the lesson failed to arrive:**
1. This card was returned by the author's own KB search hours earlier in the same session and
   not opened — the second such miss that day (cf. OL-310 Part C's provenance note). **A card
   surfaced and unread is indistinguishable in outcome from a card that does not exist**, which
   is why "the KB already had it" must be treated as a mechanism defect, not a personal lapse.
2. The report was framed as *"not saying you are wrong"* while proposing a cause for someone
   else's artifact. Politeness does not lower the evidence bar — **a hedged wrong cause is
   still a wrong cause**, and hedging makes it likelier to be waved through.

**Cross-ref**: OL-297 (a check aimed at the wrong mechanism), OL-305 (the control-side
sibling), OL-307 (the same laxity inside a correction), OL-299 (a value in the right format
from the wrong object), OL-310 Part C (a real gap in a DIFFERENT tool, correctly cited — the
contrast that makes this one legible). backend=all.
