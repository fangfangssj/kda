---
type: Runbook
title: "A control, a baseline and a reference path are themselves under test — the group that passes for free is usually the group that measured least"
description: "A control, a baseline and a reference path are themselves under test — the group that passes for free is usually the group that measured least"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-305"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-305 — OL-305: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (verification design)。
- 验证来源：two independent workstreams, 2026-07-30/31 -- a 56-case precision grade (host-side arithmetic, no device claim) and a collective-communication numerical fixture (on-device, 2 ranks)。

**OL-305** — **A control, a baseline and a reference path are themselves under test: the group that passes for FREE is usually the group that measured LEAST.** Three instances, two workstreams (2026-07-30/31). (a) 12 of 56 cases had a zero denominator and were assigned pass by the comparator's zero-denominator rule, not measured — and under an RMSE-style absolute criterion those same 12 invert from the most permissive to the **tightest** constraint in the set (~2e-9 L2 budget), because the relative comparator FLOORS error below `thr*1e-3` while RMSE has no floor ⇒ report **N measured + M rule-assigned + non-finite count on its own line**, never one fraction. (b) A control run to exclude tag-contamination returned **100 % bit-exact / 0 ULP** on a cross-arch reduction whose floor is 1–2 ULP — identity experts + unit weights make the K contributions identical, so `x+x=2x` is exact and **the rounding reduction was never traversed**; the author had caught that same degeneracy in their own fixture hours earlier, having applied four degeneracy checks to the test and zero to the control. (c) A `max\

- Former A5Ops ID: `OL-305`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (verification design)`
`verified_on: two independent workstreams, 2026-07-30/31 -- a 56-case precision grade (host-side arithmetic, no device claim) and a collective-communication numerical fixture (on-device, 2 ranks)`
**Principle**: reviewers spend their degeneracy budget on the thing being measured and
almost none on the apparatus measuring it. But a control is a program too, and the way it
earns its clean result is often *by not exercising the path under study*. Two symptoms name
the same failure: a case that passes without being measured, and a control that returns a
perfect score. Both are read as reassurance; both are the absence of a measurement.
**Instance A — cases assigned a verdict by a rule rather than by measurement.** In a 56-case
grade, 12 cases had a zero denominator, and the comparator's documented zero-denominator
branch assigns `1.0` (pass) when both sides are zero. Reported naively that is "56/56". Under
the relative metrics those 12 are the *easiest* cases in the set. Switch the criterion to an
RMSE-style absolute metric and they invert to the **tightest** constraint in the whole set
(an L2 budget around 2e-9), because the relative comparator **floors** any error below
`thr*1e-3` while RMSE has no floor. The same 12 cases are simultaneously the most permissive
and the most demanding, depending only on which criterion is applied — so "they pass" was
never a fact about the kernel.
⇒ Reporting form that survives this: **N measured + M rule-assigned, with the count of
non-finite ratios on its own line.** Never a single fraction.
⚠️ **Version scope on instance A — name the comparator, do not name "the standard".** The
relative-metric and RMSE-style comparators used above are **specific comparator
implementations we had vendored, not a claim about which criterion is currently
authoritative**; the upstream precision-standard scripts are under active churn (a
double-benchmark ratio comparator was deleted upstream 2026-07-13, and the relative-metric
comparator was subsequently replaced by a mixed `atol + rtol*|golden|` / matched-ratio /
`max(fixed, 32*ULP)` check). **The transferable content of this entry is criterion-independent**:
a floored relative metric and an unfloored absolute metric rank degenerate inputs in *opposite*
order, so any change of comparator re-ranks which cases are the binding constraint — and the
free-passing group must be re-measured, not carried over. Which comparator is authoritative for
a given deliverable is a **pinning decision** (pin a commit and record it, or track upstream and
re-baseline on every change); it is not settled by this card.
**Instance B — a control whose cleanliness came from a degeneracy the author had already
found once.** To exclude the hypothesis "my tag values are contaminating the payload", the
author ran a control with an identity mapping and all-ones weights. Result: **100 % bit-exact,
0 ULP**, on a cross-architecture reduction where 1–2 ULP is the floor. The explanation is the
control's construction: with identity experts and unit weights the K contributions are
*identical*, so the reduction is `x + x = 2x`, and multiplication by two only changes the
exponent — exact in binary floating point. **The reduction path with rounding was never
taken.** The author had caught this exact degeneracy in their own test fixture hours earlier;
they applied four degeneracy checks to the test and zero to the control, because "control"
did not read as something that could be wrong.
⇒ Scope discipline that follows: state what the control **can** and **cannot** answer. That
one answered "do the tags contaminate the payload" (yes, they did not) and could not answer
"is the measurement path correct", because it never traversed a rounding reduction.
**Instance C — an aggregate whose region boundary was assumed, on both sides of the
argument.** A non-determinism claim was supported by `max|a-b|` taken over a whole buffer
whose declared length was the *worst-case capacity* (it equalled `S*K*world_size`), not the
number of rows actually written. The challenge ("that maximum may be measuring never-written
padding") was answered by splitting at an assumed live length — which imported the same
unverified boundary into the rebuttal. **The claim was then rescued by a boundary-independent
argument**: two tensors returned from the same call were compared over *the same* interval,
and one was bit-identical across runs while the other was not. Whatever the boundary is, it
applies to both tensors equally.
⇒ **The general move: when a region boundary is unmeasured, prefer a comparison in which the
unknown boundary cancels.** Also: `max|a-b|` between two runs of *unrelated garbage* is not
an error magnitude — it is seed-dependent and has no physical meaning; report the qualitative
fact (same input, same process, two different returns) instead of dressing it as a deviation.
**Rule**: before accepting any clean result — a passing control, a green baseline, a
free-passing case group — answer in writing: (1) which code path did this traverse? (2) which
of its inputs are degenerate (equal, zero, unit, exactly representable)? (3) if the defect
were present, what would this have printed? A control that cannot fail is not evidence, and
"easily green" is a prompt to ask **what was measured**, not to record a pass.
**Cross-ref**: OL-297 (discriminating power — same family, stated for the check rather than
the control), OL-298 (the instrument is not the build under study), OL-304 (an unfaithful
oracle flatters the kernel and can mask a real loss), OL-262 (coverage gaps surface
truth-source bugs). backend=all.
