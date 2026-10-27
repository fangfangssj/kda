---
type: Runbook
title: "Probe a (criterion, case-set) pair with a zeros kernel — the score a do-nothing implementation achieves is that pair's vacuity rate"
description: "Probe a (criterion, case-set) pair with a zeros kernel — the score a do-nothing implementation achieves is that pair's vacuity rate"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-309"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-309 — OL-309: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (precision-criterion / case-set design)。
- 验证来源：FlashKDA A5 forward, 56-case bf16 grade, 2026-07-31 -- host-side arithmetic over a recorded per-case grade artefact; no device run, no card。

**OL-309** — **Probe a (criterion, case-set) PAIR with a ZEROS KERNEL: the score a do-nothing implementation achieves IS that pair's vacuity rate** — arithmetic over the recorded golden, no device, no rebuild (`

- Former A5Ops ID: `OL-309`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (precision-criterion / case-set design)`
`verified_on: FlashKDA A5 forward, 56-case bf16 grade, 2026-07-31 -- host-side arithmetic over a recorded per-case grade artefact; no device run, no card`
**Principle**: a precision criterion is meaningless in isolation — only the pair *(criterion, case set)* has a discriminating power. The cheapest way to measure that pair is to ask what score an implementation that **computes nothing** would get. Feed a zeros output through the real grader: every case it passes is a case the pair cannot fail, regardless of what the kernel does. That number is the pair's **vacuity rate**, and it belongs in every precision report next to the pass rate.
**The instrument** (no device, no kernel, no rebuild — it is arithmetic over the recorded golden):
```
zeros kernel  =>  |err| == |golden|  exactly, elementwise
mixed-tolerance leg : element matched iff |g| <= atol + rtol*|g|  <=>  |g| <= atol/(1-rtol)
                      case fully vacuous iff truth_absmax <= atol/(1-rtol)
max-abs-error leg   : max_abs_error == truth_absmax
                      case vacuous iff truth_absmax <= max(fixed_limit, 32*ULP)
```
Omitting `rtol` only ever makes vacuity *harder* to claim, so an rtol-free bound is conservative and safe to publish.
### ⛔ The conjunction trap — the mistake this card exists to prevent
A criterion of the form `is_pass = matched_ratio >= R AND max_abs_error <= L` has **two vacuity rates, and they can differ by 4×**. Measured on one 56-case bf16 set with the vendor's real bf16 row (`atol = rtol = 2^-6 = 1.5625e-2`, `max_abs_error_limit = 1e0`, `R = 0.99`):
```
loose leg   max_abs_error <= 1e0                     -> 52/56 vacuous (92.9%)
tight leg   99% of elems with |g| <= 1.5873e-2       -> >=13/56 vacuous (23.2%)
AND (the actual criterion)                           -> >=13/56 (23.2%)
   of those 13, cases currently FAILING the old gate -> 0
```
⇒ **In a conjunction the loose leg grants nothing; the tight leg binds.** Reporting the loose leg's 92.9% as "a do-nothing kernel passes 92.9%" is a statement about one leg masquerading as a statement about the criterion — it overstates vacuity by 4× and invents a hole that does not exist. **Always report per-leg AND name which leg binds.** The same discipline applies in reverse for a disjunctive criterion, where the loose leg is the only one that matters.
### ⚠️ Second trap, mine: a provenance tag is not a confidence downgrade
The 4× error above happened because an `atol` was **borrowed from a neighbouring, differently-scoped standard** (`1e-1` from a Triton-verifier variant) while the real value was `2^-6 = 1.5625e-2` — 6.4× smaller. The borrowing **was** labelled *"substituted, recompute when the real table arrives"* — and the conclusion built on it was still put in the **headline**, with an alarm marker.
```
provenance tag in parentheses   -> nobody reads it
number in the headline          -> everybody reads it, and forwards it
observed next hop: a peer substituted the borrowed value into their OWN fixture as a known input
observed next hop: the coordinator forwarded the derived figure upward as a priority finding
=> a value correctly marked PROVISIONAL became "known" in one hop and "worth escalating" in two,
   with nobody misrepresenting anything
```
⇒ **Tag the CONCLUSION's reliability, not just the input's provenance** — and keep a provisional-input conclusion out of the headline entirely. A number's prominence, not its footnote, determines how it travels. Corollary observed the same session: a figure can be promoted for **rhetorical** properties ("it is short, quotable, needs no explanation") which are strictly orthogonal to being correct — a wrong number can outscore a right one on every one of them.
### ✅ What the probe legitimately established on this case set
```
truth_absmax over 56 cases : min 1.85e-6 | p25 0.0366 | median 0.0581 | p75 0.135 | max 744
only 4/56 cases exceed 1e0 (209 / 266 / 640 / 744)
element-level degenerate LOWER BOUND (pooled): 495,067 / 763,392 = 64.85%
   derivation: the artefact records n_sv = count of NONZERO elems with |g| < 2^-8 = 3.906e-3;
   since atol (1.5625e-2) > 2^-8 with 4.0x margin, EVERY such elem satisfies |g| <= atol
   => a bound that does NOT depend on the threshold table, only on the inequality atol > 2^-8
   => it is a LOWER bound twice over: zeros are excluded from n_sv, and the real
      cut (1.5873e-2) is 4.1x wider than 2^-8
   => FALSIFICATION CONDITION, stated so the next reader need not redo it:
      upstream dropping bf16 atol to 2^-8 or below (a 4x cut) is what would void it
per-case: median sv_frac 70.0% | 46/56 cases >50% | 13/56 >90% | at least one at 100%
```
**Dilution consequence** — a nominal `matched_ratio >= 0.99` is not a fixed strictness. With degenerate fraction `f`, only `(0.99 - f)/(1 - f)` of the *meaningful* elements must pass:
```
f = 64.85% (pooled bound) -> effective 97.16%      f = 90% (13 of 56 cases) -> effective 90.00%
f = 69.97% (median case)  -> effective 96.67%      f = 100%                 -> UNCONDITIONALLY TRUE
```
⇒ **Two runs both reporting "0.99" can embody entirely different strictness, and the report cannot show it.** So `matched_ratio` is not a comparable quantity unless published with the degenerate count **and** its fraction, at **both granularities**: element-level (how polluted the ratio is) and case-level (how many cases should not count toward a score at all).
⇒ Count degenerates as `#{ |golden| <= atol }`, **not** `#{ golden == 0 }`. The dangerous element is a *nonzero* one that still sits under an absolute tolerance — it looks like real data and cannot fail. A `golden == 0` counter puts it in the "meaningful" denominator and thereby hides part of the blind spot it was written to expose.
### ⭐ The finding underneath: one physical property, opposite failures under the two criterion families
```
FlashKDA output magnitudes are small (median truth_absmax 0.058)
  RELATIVE criterion (max/mean of |err|/(|g|+eps)) -> small |g| AMPLIFIES  -> 37/56 MARE fails -> TOO STRICT
  ABSOLUTE criterion (atol / max_abs_error)        -> small |g| IMMUNISES  -> >=64.85% of elems
                                                       cannot fail        -> TOO LOOSE
=> neither family measures this operator; they fail in opposite directions
=> the independent variable is the CASE SET's magnitude range, not the criterion
```
⇒ **Therefore the actionable lever is the case set, not the criterion.** A criterion is external and tracks upstream (which changed twice in one month here); the case set is ours. Extend it to cover magnitudes the tolerance cannot swamp, or make the limit adapt to each case's `truth_absmax`. Choosing *between* criterion versions only selects a failure direction.
**Rule**: (1) publish a zeros-kernel vacuity rate beside every precision score, **per leg**, naming the binding leg; (2) publish the degenerate count and fraction at element and case granularity; (3) treat a comparator change as invalidating the free-passing group — **re-measure it, never carry it over**; (4) when a criterion and a case set disagree about what is hard, suspect the case set's dynamic range first.
**Degenerate-reference instance** (4_GroupedMatmulSwigluQuant, Ascend950PR_9579 / CANN 9.1.0.B150, 2026-09-07): under the operator's older contract the *reference itself* was degenerate on the whole denominator — output 0 all-zero and output 1 all non-finite on 60/60 cases (fp16 post-activation overflow, PB-76). Measured with the bound product comparator: a NULL candidate returning only correctly-shaped zeros and Infs was accepted 60/60, while a constant-127 fill was accepted 0/60 — the gate was live but could not reject *the* degenerate constant. After the contract's source changed, a null candidate and a `y+2` perturbation were rejected and `y+1` accepted (the int8 `|diff| <= 1` criterion itself). The record's own day-one check: run the NULL candidate early (correctly-shaped constants that compute nothing) against the bound comparator; here it would have exposed the vacuity before any tuning. A zeros-only probe would not have shown it on output 1, whose golden is itself non-finite. Evidence (agent-team-wiki @5433fae0): `projects/op4_4/DECISION_20260907T0400Z_DS_OP4_4_GATE_HAS_NO_DISCRIMINATING_POWER.md`, `projects/op4_4/RULING_20260908T1904Z_DS_OP4_4_DENOMINATOR_BOUND_AND_CLOSED.md`.
**Cross-ref**: OL-305 (the general form — the free-passing group measured least; instance A there is this operator's relative-criterion side), OL-297 (discriminating power: what a check prints with the defect absent vs present — the zeros kernel is that question made executable), OL-304 (oracle dtype design), OL-306 (a borrowed value's relevance is assumed, not established). backend=all.
