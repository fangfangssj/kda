---
type: Runbook
title: "Cheap and binary is not discriminating power — write down what a check prints with the defect absent and with it present; same string means the check measures nothing"
description: "Cheap and binary is not discriminating power — write down what a check prints with the defect absent and with it present; same string means the check measures nothing"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-694"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-694 — OL-297: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all。
- 验证来源：four instances in one night, 2026-07-30/31, across two independent workstreams (FlashKDA A5 forward + a peer harness workstream) -- host-side and offline evidence only, no device claim in this card。

**OL-694** — **Cheap and binary is NOT discriminating power: before trusting a check, write down what it prints with the defect ABSENT and with it PRESENT — same string ⇒ the check measures nothing, regardless of cost or crispness.** Distinct from CAND-A5OPS-0016: a guard-byte canary passes both calibration controls (poked ⇒ FAIL, clean ⇒ PASS) and is STILL void when the phenomenon is a race, because the inputs it separates are not the inputs in question. 4 anchors in one night, 2 workstreams: guard bytes vs a race (PASS would have read as "overrun excluded"); a gate printing `0/0/0` with the patch active AND reverted; a test case that never traversed the patched path; a trade-off sweep scored on the conjunct 54/56 cases already passed (`fail_caused_by_mere_only: 0`) while all 38 failures were on the other one. **Two controls to copy**: read the sentinel fill back BEFORE measuring (any test depending on a setup step must assert that step succeeded), and poke the defect in to prove the comparator can fail AND localise (4 bytes ⇒ exactly 2 flagged guards at the right offsets).

- Former A5Ops ID: `OL-297`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all`
`verified_on: four instances in one night, 2026-07-30/31, across two independent workstreams (FlashKDA A5 forward + a peer harness workstream) -- host-side and offline evidence only, no device claim in this card`
**Principle**: the two properties that make a check attractive to propose — it is
**cheap** and it returns a **binary verdict** — say nothing about whether it can
separate the hypothesis from its alternatives. Discriminating power is a property of
the check's mechanism *against the defect's mechanism*, and it is independent of cost
and of output crispness. A cheap binary check aimed at the wrong mechanism emits a
**true** statement with no bearing on the question, and that statement gets read as an
answer because it looks like one.
**The required question — ask it before building, not after**: *what does this check
output in the world where the defect is ABSENT, and what does it output in the world
where it is PRESENT?* Write both down. If they are the same string, the check is void:
do not run it, because its output is indistinguishable from evidence. This is a
stronger demand than "have a control" — a control shows the instrument fires, this
shows the instrument is coupled to the mechanism.
**Four instances, one night, two workstreams:**
| # | the check | why it looked good | why it had no power |
|---|---|---|---|
| 1 | 512 B guard bytes between 40 workspace slabs, filled `0x5C`, re-read after one forward pass — to catch an out-of-bounds write | cheap (+20 KiB, one build), binary (`PASS_GUARDS_INTACT` / `FAIL`), correctly implemented, five built-in controls | the phenomenon was subsequently characterised as a **race**, not an overrun. Guard bytes sit intact under a race and print PASS — and that PASS would have been read as "an overrun is excluded" |
| 2 | a strict verification script whose summary counters were required to read `0/0/0` | one command, three integers, unambiguous | it printed `0/0/0` **with the patch under test active and with it reverted** — identical output in both worlds |
| 3 | a named test case cited as validation of a patch | the case existed, ran, and passed | the case **never traversed the patched code path** — it measured code unrelated to the change while being counted as evidence for it |
| 4 | an fp32-promotion trade-off sweep scored on `mere` | a real device sweep over a real promotion ladder | `mere` is one conjunct of a two-part gate and **54 of 56** cases already passed it (`mere_pass: 54`, `fail_caused_by_mere_only: 0`); all 38 failures were on `mare`, whose per-case values had been printed hours earlier in an unrelated grading run. The curve could only move the conjunct that was not failing |
**Instance 1 is NOT a calibration failure, and that distinction is the point.** The
guard canary passes both of CAND-A5OPS-0016's controls: poke a byte and it reports FAIL, leave it
alone and it reports PASS, so it demonstrably separates two known-different inputs. It
was still void, because **the inputs it separates are not the inputs in question**.
Calibration proves the instrument fires; it does not prove the instrument is coupled to
the defect's mechanism. Both properties are required and neither implies the other.
**How instance 1 was actually settled — and the settling evidence was cheaper than the
instrument:** (a) across four runs of the same `.so` the outcome set was **two-valued**,
each value reproducible to 17 significant digits, where memory corruption gives an
unbounded spread; (b) changing the workspace allocation from `torch.empty` to
`torch.zeros` changed **nothing** — same differing case, identical max abs diff
`9.67979e-03` — so nothing was reading stale bytes; (c) the affected case was the suite's
**only cold launch**. A small discrete outcome set that **survives zero-filling** is the
signature of a data race or a missing handshake. The specific unsynchronised access is
**NOT localised** — this card claims the class, not a line.
**The two controls worth copying** (both implemented in the instance-1 instrument, and
worth keeping even though the instrument itself was cancelled):
1. **Verify the instrument fired.** After filling with a sentinel, **read the fill back
   and compare it to the sentinel BEFORE the measurement runs**, and abort if it did not
   land. A silently-failed fill leaves every guard reading "intact" and makes the whole
   test vacuous. Generalised: **any test that depends on a setup step must assert that
   step succeeded**, or a setup failure becomes indistinguishable from a passing result.
   Same trap as an all-zero `lo` slab looking like a working split.
2. **Prove the comparator can fail.** Deliberately introduce the defect the check exists
   to find — poke bytes, corrupt a value — and require the comparator to report it **and
   localise it correctly**. Exercised offline here, no card needed: on a clean synthetic
   snapshot 0 guards corrupt; with 3 bytes poked at one slab's guard start and 1 byte at
   +300 in another's, exactly **2** guards flagged, at the right offsets. A comparator
   whose failure branch has never executed is an untested branch on the critical path.
**Where to expect this class**: any canary / guard / sentinel; any grep-as-negative-
assertion; any "must print 0" gate; any test cited for a patch without a reachability
check on the patched path; any sweep or trade-off curve over one conjunct of a
multi-part gate (check WHICH conjunct fails first — the values are usually already in
an earlier run's output).
**Cross-ref**: CAND-A5OPS-0016 (controls prove the instrument fires — necessary, not sufficient;
this card is the mechanism-coupling half), OL-296 (reproduce the authority's number
before its decomposition means anything), OL-298 (the same instrument perturbed its
subject's layout), OL-207 / OL-208 / OL-209 (refutation discipline once an instrument is
known good).
---