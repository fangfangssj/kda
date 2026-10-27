---
type: Runbook
title: "When a spec ships both prose and an implementation, cite the implementation's file:line — and a gate can be unsatisfiable, which is zero measurement, not strictness"
description: "When a spec ships both prose and an implementation, cite the implementation's file:line — and a gate can be unsatisfiable, which is zero measurement, not strictness"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-310"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-310 — OL-310: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (spec citation + gate authoring)。
- 验证来源：the cannbot/opbase precision-standard alignment, 2026-07-31 -- one gate accumulated THREE independent defects found by three people inside half an hour; source-verified, host-only。

**OL-310** — **(A) When a specification ships BOTH prose and an implementation, cite the implementation's `file:line`; prose is for LOCATING only. (B) A gate can be UNSATISFIABLE — that is not strictness, it is zero measurement.** Both are defects in the gate's or spec's own TEXT, and both survive review for the same reason: (A) reads as *authoritative* (official prose), (B) reads as *rigorous*. **(A)** opbase's §2.2 prose says `"1e-0 or 32 * ULP"` while the cannbot checker does `max(fixed_limit, ulp_limit)` (`mixed_tolerance_check.py:96-97`, **cited as the coordinator's reading — that file is ABSENT from our vendored tree**) — "or" reads as either-one-may-be-used, `max()` means ALWAYS THE LOOSER, and the two readings give different verdicts on the same tensor. The same document bit the same reader **twice, in different shapes**: first reading the table's VALUES and verifying column alignment but not how they COMBINE (missing that the criterion is an **AND**), then reading the PROSE's wording while the CODE produces the verdict. Common factor: **the text read is not the text that executes, and both are official.** Note the first bite's shape — a verification step WAS performed and covered exactly the part that could not go wrong, so afterwards it reads as "already checked". **First-hand anchor, read in-repo**: `cannbot_standard/PROVENANCE.md` says "VERBATIM COPY … Commit 7d13b6a (2026-06-12) … DO NOT edit these files — they are the upstream standard", then enumerates as THE criteria two scripts upstream has since **deleted and replaced** ⇒ **"verbatim" is a claim about fidelity and says nothing about freshness — and the more emphatic the fidelity claim, the more settled the file reads.** Record such prose-vs-code discrepancies IN the provenance file. **(B)** A gate read *"the precision criterion's self-check must REJECT the known-degenerate integer fixture"* — but integer inputs are exact in bf16, so error is 0, so the criterion **necessarily ACCEPTS**; the instrument that rejects that fixture is a rounding-rate (fixture-quality) check. **Two instruments had been written as one gate.** Implemented literally there are only two outcomes — the implementer silently reinterprets it (gate did nothing) or manufactures a fake pass — **and both read as "gate satisfied" in the PR**, which is worse than no gate. This is the **mirror of OL-297**: there the *check* cannot fail, here the *requirement* cannot be met, and it is harder to catch because **impossibility presents as strictness — strict and discriminating are different properties.** **Authoring rules**: write what satisfying the gate concretely looks like and confirm that state is REACHABLE before asking if it is strict enough; **name the instrument that produces the verdict**; a discriminating-power gate needs TWO ARMS (one accepted, one rejected, printing differently). **Where review effort belongs**: this one gate accumulated **three independent defects, three finders, ~30 minutes**, no two the same (wrong instrument / single-armed / unsatisfiable) — gate TEXT deserves the same adversarial reading as gated code, and the cheapest form is requiring every gate to state its defect-present vs defect-absent output: **a gate that cannot describe both is not yet a gate.** Related practice: when a session retires numbers, gate KB diffs on the retired figures — **a number entering the KB sheds the "retired" context it carried in conversation** — with the criterion "if present, must sit beside the correct value and be marked wrong", not "reject on appearance".

- Former A5Ops ID: `OL-310`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (spec citation + gate authoring)`
`verified_on: the cannbot/opbase precision-standard alignment, 2026-07-31 -- one gate accumulated THREE independent defects found by three people inside half an hour; source-verified, host-only`
**Why these two live on one card**: both are defects in the **gate's or the spec's own text**, not in the
work being gated — and both survive review for the same reason. The first reads as *authoritative*
(it is official prose); the second reads as *rigorous* (it is a strict-sounding requirement). Neither
looks like something that needs checking, and a reviewer's attention goes to the implementation.
### Part A — two official texts, and only one of them decides
A vendor standard that publishes a human-readable table **and** an executable checker has **two official
texts**. The verdict comes from the code. The prose is the **quotable** half, so the prose is what
propagates into task specs, task briefs and downstream cards.
```
opbase prose (§2.2 table)        "1e-0 or 32 * ULP"
cannbot implementation           mixed_tolerance_check.py:97  max_abs_error_limit = max(fixed_limit, ulp_limit)
                                                        :96  ulp_limit = 32 * _ulp_at_one(dtype)
=> "or" reads as EITHER-ONE-MAY-BE-USED (an implementer may pick the smaller, i.e. the stricter)
   max() means ALWAYS THE LOOSER OF THE TWO
=> not a paraphrase difference: the two readings give different verdicts on the same tensor
```
⚠️ **Attribution, because this card is partly ABOUT citing precisely**: those two implementation line
numbers are the **coordinator's** reading, not this author's — `mixed_tolerance_check.py` is **absent from
our vendored tree**, which still sits at the pre-migration commit (see the verified anchor below). Cited
as reported, not as re-verified.
**A first-hand anchor for the same point, read directly in this repo** — the vendoring provenance file
guarantees the wrong property:
```
src/scripts/orchestrator/precision/cannbot_standard/PROVENANCE.md
  "VERBATIM COPY ... Commit: 7d13b6a (2026-06-12) ... Copied verbatim per owner directive
   ('完全照抄 cannbot'). DO NOT edit these files — they are the upstream standard."
  and it then enumerates, as THE criteria:
     商用 双標杆Ratio  scripts/mare_mere_rmse_ratio.py
     生态 单標杆Threshold  scripts/mare_mere_threshold.py
  vendored scripts/ actually present: mare_mere_rmse_ratio.py, mare_mere_threshold.py, ...
                                      and NO mixed_tolerance_check.py
```
⇒ Upstream **deleted** the ratio script and **replaced** the threshold script in the interval. So a file
that says *"do not edit, this IS the upstream standard"* now describes two retired criteria as current,
and does so in the most authoritative-sounding wording available. ⛔ **"Verbatim" is a claim about
fidelity; it says nothing about freshness — and the more emphatic the fidelity claim, the more the file
reads as settled.**
**The same document bit the same reader twice, in two different shapes:**
```
1st  read the table's VALUES and verified column alignment -- but not how the values COMBINE
     => missed that the criterion is an AND, and briefed a task that implied either leg could pass
2nd  read the PROSE's wording while the verdict is produced by the CODE
     => copied "or" where the implementation does max()
common factor: the text that was read is NOT the text that executes -- AND BOTH ARE OFFICIAL.
```
⇒ Note what makes the first one nasty: a **verification step was performed** (column alignment,
deliberately, because a mis-read table row was a known hazard) and it covered exactly the part that could
not go wrong. **A careful-looking action can precisely miss the risk, and afterwards it reads as
"already checked".**
⇒ **Rule**: cite the implementation's `file:line` for any criterion, threshold or combining rule; use the
prose only to *locate* the right code. When vendoring such a standard, record the discrepancy **in the
vendoring provenance file** — a note saying "prose says `or`, implementation is `max()`, code governs",
with both line numbers. A `PROVENANCE.md` reading "copied verbatim from upstream, do not edit" buys
**fidelity, not correct reading**, and the next person aligning from the document hits the same trap.
### Part B — a gate that cannot be satisfied measures nothing, and reads as rigour
Gate as written: *"the precision criterion's self-check must REJECT the known-degenerate integer
fixture, proving it discriminates."* Integer inputs are exactly representable in bf16, so a correct
kernel's error is **0**, so every element is matched: **the criterion necessarily ACCEPTS.** The
instrument that rejects that fixture is a *rounding-rate* check — a **fixture-quality** instrument, not a
precision criterion. Two instruments had been written as one gate.
```
If implemented literally, only two outcomes exist:
  (a) the implementer silently reinterprets it as something achievable => the gate did nothing
  (b) the implementer manufactures a fake pass
BOTH read as "gate satisfied" in the PR. That is worse than a gate that is simply absent.
```
⇒ This is the **mirror** of "a check that cannot fail measures nothing" (OL-297): here the *requirement*
is the impossible half. **It is harder to catch, because impossibility presents as strictness.**
⛔ **Strict and discriminating are different properties**, and a gate can max out one while scoring zero
on the other.
⇒ **Rules for authoring a gate**: (1) write down what *satisfying it* looks like concretely, and confirm
that state is **reachable**, before asking whether it is strict enough; (2) **name the instrument that
produces the verdict** — the conflation above was two instruments sharing one sentence; (3) a
discriminating-power gate needs **two arms** (something accepted *and* something rejected, printing
differently) — one arm admits a reject-everything instrument (OL-297).
### What the incident says about where review effort belongs
One gate, **three independent defects, three finders, ~30 minutes**, and no two found the same one: wrong
instrument (carrier), single-armed control, and unsatisfiable-as-written. The author was neither careless
nor unskilled — they wrote it while thinking "prove discriminating power" and produced a sentence that
could not do that. ⇒ **Gate text deserves the same adversarial reading as gated code**, and the cheapest
form is to make each gate state its **defect-present vs defect-absent output**: a gate that cannot
describe both is not yet a gate.
**Related practice from the same session** (the coordinator's, worth copying): when a session voids
numbers, add a merge gate that greps the retired figures in any KB diff — because **a number that enters
the KB sheds the "retired" context that travelled with it in conversation.** The criterion is not
"reject on appearance" but *"if present, it must sit beside the correct value and be marked as wrong."*
**Cross-ref**: OL-297 (a check whose output is identical with the defect present and absent — Part B is
its mirror on the requirement side, and its two-control rule is what Part B's rule 3 inherits), OL-299
(a value in the right format from the wrong object — Part A is the spec-text instance), OL-305 (the
free-passing group), OL-309 (the conjunction trap this same standard produced: the AND missed in bite #1).
backend=all.

### Part C — a gate that CAN be satisfied but degrades silently when it cannot RUN

Parts A and B cover a gate that is mis-aimed or unsatisfiable. This is the third shape, and it is the
one that survives review, because the gate demonstrably works when you test it: it emits a **normal
verdict** when it fails to run, and that verdict is **byte-identical** to a real one.

**Measured (2026-08-01, workspace-triage tooling, three agents' disks).** A scanner classified whether a
tensor dump is regenerable. Its seed lookup resolved imports relative to the repo, so a copy of the
script run from `/tmp` resolved nothing and every operator came back `NO_SEED`:

| | `REGENERABLE` | `NO_SEED` |
|---|---|---|
| run inside the repo | 38736.9 MB / 44 ops | 0 MB |
| same script, same commit, run from `/tmp` | 6411.9 MB / 9 ops | 32325.0 MB / 36 ops |

⇒ **32.3 GB changed verdict on the location of the script**, 83% of the bulk misjudged. Not a wrong
answer — a *confident* one.

⭐ **The load-bearing observation is not the tonnage, it is who got blamed.** Two agents independently
obtained the false reading and each was about to report that a (correct) fix "did not work on my tree";
one stated they had the false number in hand and only avoided sending it because the other's warning
arrived first. ⇒ A failure mode that makes two independent readers accuse the same innocent change
cannot be defended against by "remember to run it from the right place".

**Rules this adds to Part B's list:**
1. **Separate "could not check" from "checked, found nothing" as distinct verdicts.** Here: refuse to
   start when the resolution root is unreachable, and per-item emit `UNRESOLVED_IMPORT` rather than the
   negative verdict. A negative that also means "I was blind" is unusable (cf. OL-311).
2. **Emit the verdict's provenance, not just the verdict** — which paths were consulted, which were
   followed, where the tool believes it is standing. This is what let both readers locate their own
   fault in minutes instead of filing a bug against someone else's work.
3. **Which direction of error is "safe" is a property of the CONSUMER, not of the check.** The same
   scanner had both: a false `AMBIGUOUS` (over-reading) made it cry wolf, so the one real ambiguity
   would be ignored; a false `NO_SEED` (under-reading) reads as a *fact* rather than as uncertainty, so
   nobody re-checks it. **Neither direction is the conservative one** — "when unsure, fail closed" is
   not a general answer, and picking a direction requires naming who acts on the output.
4. **An error message that does not say what to do next is barely better than the wrong answer.** The
   first version of this same guard raised `IndexError` before it could print the flag that fixes it.

⇒ **Provenance note on how this entry was written**: the author re-derived rules 1–4 while building the
tool, then found them already stated in Parts A/B of this card, having skipped the KB load that
`CLAUDE.md` requires before op-gen-adjacent work. The re-derivation was not free and the lesson would
have stayed in a private note. ⇒ **Prefer adding evidence to an existing card over opening a parallel
one**; and treat "the KB already had it, I just didn't look" as a mechanism defect — a brief should be
required to *report which KB entries it loaded*, since an unreported load is indistinguishable from none
(which is rule 2, applied to ourselves).
