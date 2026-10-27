---
type: Runbook
title: "Arms timed sequentially inside one per-case block carry an ordering artefact of several percent — same-NPU, same-session, back-to-back is necessary and not sufficient; run a null arm at the position of the arm you care about"
description: "Arms timed sequentially inside one per-case block carry an ordering artefact of several percent — same-NPU, same-session, back-to-back is necessary and not sufficient; run a null arm at the position of the arm you care about"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-450"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-450 — OL-450: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B150; scope=paired device-time A/B methodology (torch.npu.Event, within-block arm comparison); op_class=all。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0.B150 (11_GroupedMatmulSwigluQuantV2, 2026-09-11)。

**OL-450** — **arms timed sequentially inside ONE per-case block carry an ordering artefact of several percent; "same-NPU, same-session, back-to-back" (OL-27) is necessary and NOT sufficient.** Always run a NULL arm — the identical configuration twice — and place it at the position of the arm you care about.

- Former A5Ops ID: `OL-450`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B150; scope=paired device-time A/B methodology (torch.npu.Event, within-block arm comparison); op_class=all`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B150 (11_GroupedMatmulSwigluQuantV2, 2026-09-11)`

**Problem**: three independent A/B designs for the same candidate each produced an apparent win, and each was destroyed by a *different* ordering artefact. None of the three was detectable from the arm data alone.

1. **The alternation key WAS the design.** Arm order was keyed on `case_index % 2` — and the contract cycles its `(dequant_mode, quant_mode, group_list_type)` tuple on `case % 4`, so "arm order" and "parameter group" were literally the same partition. The reported 4.4% "order effect" was the parameter groups wearing an order label. Removing it cost one repeat with every case's arm order flipped; arm order then measured GM **1.011** while run-to-run dispersion was **6.6%**.
2. **A null arm won every window.** In a `base_n`/`base_k` sweep, `n256 k256` beat the default in all three windows by **+2.01%** — and `pybind11.cpp` reads `baseN = (baseNReq > 0) ? baseNReq : GMMSQ_BASE_N` with `GMMSQ_BASE_N = 256`, so that arm *was* the default, byte for byte. A rule of "wins in every window" is satisfiable by a configuration that changes nothing.
3. **Position, and then adjacency.** Four arms in fixed order came out monotone in position (1st `0.978` -> 4th `1.108`); a null arm identical to arm A gained **+3.87%** purely by being second. **Rotating the arm order did not fix it**: rotation permutes position and PRESERVES adjacency, so the null still followed A in every case. After rotation the positional trend fell to +2.02% but A and the null still differed **+2.75%** and moved together window to window (+6.50/+3.59/-1.61 against the claim's +9.70/+4.72/+0.12).

**Rules**:
1. Every A/B carries a **null arm**: the same configuration measured twice. A claim counts only if it beats the null, not if it beats the baseline.
2. **Place the null where the claim sits.** A control one slot earlier controls nothing — this is what defeated design 3.
3. **A rotation is not a randomisation.** If adjacency must be broken, use a randomised or Latin-square order and say which.
4. Before keying an alternation on the case index, tabulate each case's parameters against the proposed key and check the partition is not the same one.
5. **The stability of a repeated arm is not the resolution between arms.** The default arm here was steady to **0.12%** across windows while two identical configurations differed by **3.37%**.
6. When the margin is smaller than the artefact, the honest output is `UNMEASURED` plus a statement of the method's limit — not another draw.
7. **Pairing is not a null.** An ABBA window (or any bracketing window that times the same bytes twice) removes cross-window confounding between the arms, but not the noise inside the window. Its same-bytes repeats — A2 vs A1, B2 vs B1 — measure that within-window noise, plus any effect of the two slots each repeat occupies, at no extra cost; a same-bytes difference inside the window is the quantity that bounds resolution, and the across-window stability of one arm is not (rule 5). Read every arm delta against those repeats: a delta smaller than the repeat drift leaves the comparison unresolved, and the output is rule 6's `UNMEASURED` with the repeat drift stated as the method's limit — neither an effect nor "no effect". The repeats meet rule 1's definition of a null (the same configuration measured twice) but are **not** the position-matched null that rule 2 requires: each compares one configuration with itself across two different slots, so it carries whatever position or adjacency effect separates those slots and controls nothing at the claim's position. Clearing the repeat drift is therefore necessary, not sufficient — a null placed at the claim's position (rules 1-2) is still required, and one window is still one draw (see the counter-example below).

**Measured limit for this operator**: differences below roughly **5%** between arms timed inside one block are not resolvable by this method. Settling a few-percent effect needs one arm per process, many independent windows, interleaved at the window level rather than inside a case.

**Companion measurement — the window-level dispersion**: eight independent windows of the same candidate and baseline gave mean **1.000993**, sd **0.038728**, spread **12.05%** (min 0.9665, max 1.0830), with warm-up policy worth -1.40% and arm order -2.25% — both far below the window spread. Single-window figures for that same pair ranged 0.9544 / 0.9600 / 0.9751 / 1.0004; every one was a draw from that distribution.

**Four rejected versions of the CONTROL, and then the instrument still did not qualify.**
A carryover-balanced schedule (Williams design over the arms, the complete cycle inside every
case, with the baseline timed between sequences) was built to remove ordering bias. ⚠️ **It did
not produce a usable instrument, and the claim that it did is RETRACTED — see the counter-example
at the end of this entry, which is the most load-bearing part of it.** What the balanced design
does establish is narrow: the schedule executed as pinned, and position and first-order
carryover counts are balanced. Whether the null floor became usable was a separate question and
the answer measured across eight independent windows was no. Building it still cost four
rejected versions of the *control* rather than of the measurement, and each failure is a
distinct reusable trap:

1. **Verify the EXECUTED schedule, never the template.** Version 1 assigned one sequence per
   case and its verifier checked only the base sequences — it validated an object that would not
   be run, so it could not fail on the thing being measured. It also claimed strict balance that
   was arithmetically impossible (50 cases over 4 positions is 12.5). **Hash the stream the
   runner will actually execute, pin that hash, and hand the verifier THAT object.** A verifier
   fed its own reconstruction can pass while the runner executes something else.
2. **Count the boundary transitions, or remove them.** Version 2 counted only the edges inside
   each sequence. If the sequences run consecutively the real stream has `n` extra
   sequence-boundary transitions per case, and with a fixed order they concentrate on a few
   ordered pairs — the omitted edges were exactly the risk being balanced. Worse, they can be
   *unbalanceable*: with 4 arms the best achievable spread was 25. The fix was structural —
   interleave the baseline between sequences so no arm→arm boundary edge exists — after which
   the remaining transitions balance strictly and the "not an integer" problem disappears
   instead of being absorbed into a tolerance.
3. **An arm measured immediately after ITSELF is a different carryover kind.** The verifier
   crashed on a self-edge the edge model had no slot for. Self-priming and cross-arm carryover
   must not be pooled; exclude the orderings that produce it and assert zero.
4. **Passing the pre-registered gate is necessary, not sufficient — check the MAGNITUDE against
   the mechanism.** In the single window that passed, the arm differed from its control by
   +2.457%, cleared the null floor, and the magnitude checked out against the only code
   difference timed off-device (predicted 0.689 us per call, observed 0.857 us on a 34.5 us
   measurement, ratio 1.24x). **Both checks passed and the result was still wrong** — see the
   counter-example below. A magnitude check raises the bar; it does not substitute for
   repetition. And note what the host-side number is and is not: `0.689 us` is a microbenchmark
   fact about five guarded device moves, never a device-time or product gain.

**Two negative results from the same work, both worth more than the positive one.**

* **Position balance does NOT remove between-window drift.** Balancing order fixes the
  *comparison between arms inside a window*; the window-to-window dispersion (measured 12.05%
  of the GM over eight independent processes) is an orthogonal axis and is untouched by any
  ordering design. A balanced single window is still a single draw. Say which axis a claim
  rests on.
* **An arm that looks like "the floor" may be a different entry point, not a subset.** Calling
  the extension directly was read as "the wrapper's cost removed", giving an apparent +13.48%
  of available headroom. Under balance it was **−0.385%**, median −2.126%, 10/50, with three
  times the dispersion of the other arms — because it is a different call path with its own
  overhead, not the wrapped path minus the wrapper. **Before treating an arm as a bound, show it
  is a subset of what it is supposed to bound.**

**⛔ THE COUNTER-EXAMPLE, and the rule it forces: NEVER infer from ONE window, however many
checks that window passes.** This is the strongest evidence in this entry and it is evidence
against its own author.

One window of the balanced design returned a null floor of `−0.527%`, an arm difference of
`+2.457%` at 45/50 cases, a matching mechanism magnitude, a pre-flight stream hash that matched
its pinned value, and a design verifier that returned ACCEPT. Every check available passed. It
was published as "the instrument works" and "the first claim to survive both a statistical
control and a mechanism check". Eight further independent windows of the *same* schedule then
gave:

    null floor      -1.377  -1.209  -6.826  -2.416  +1.021  +1.069  +1.175  -4.372   (mean|.| 2.433%)
    valid windows   0 of 8 against the same pre-registered |floor| <= 1.0% rule
    arm difference  mean +0.220%, median -0.482%, ahead in 3 of 8
                    -> the published +2.457% was the MAXIMUM of the distribution
    the arm's GM    published 1.0406; eight-window mean 0.984904, and the published value is
                    above every one of the eight

So: **a single window passing every check it is given is still a single draw, and the published
figure will preferentially be the draw that looked worth publishing.** Selection happens at the
moment of writing, not at the moment of measuring.

**⚠️ And a second-order trap in the gate itself: a pre-registered threshold must be FEASIBLE
for the instrument.** The `|floor| <= 1.0%` rule was fixed without first measuring the floor's
standard error. Per-window s.e. turned out to be 0.95–2.10% (mean 1.414%), because one arm
measured four times inside one case varies by up to **1.5x** (median spread 8–10%, p90 90–113%).
At that resolution the gate could essentially never pass, so `0 of 8` does **not** show the
balanced design failed — it shows the design was never tested. **Measure the s.e. of the gated
quantity BEFORE fixing the bound**, and report a qualification as a distribution over
independent windows (proportion valid, floor distribution), never as one window's verdict.

**Consequence for practice**: an ordering design can only ever remove the ordering axis.
Here it left the dominant axis — within-case repetition spread, plausibly host contention
(`OL-15` / `PB-10`) — completely untouched. Qualify a new instrument with a multi-window null
study *before* measuring any candidate with it.

Evidence: 内部路径（不公开）agent-team-wiki/projects/op11-groupedmatmulswigluquantv2/ARCHIVE_20260911T0745Z_SCAN_V4_EIGHT_WINDOW_NULL_STUDY.md
(all eight windows, every INVALID reason, the floor distribution and the full retraction chain);
内部路径（不公开）agent-team-wiki/projects/op11-groupedmatmulswigluquantv2/INSTRUMENT_20260911T0710Z_SCAN_CARRYOVER_BALANCED_DESIGN.md
(design, all four rejected versions and the controls that rejected them);
内部路径（不公开）agent-team-wiki/projects/priority24/NPU_LIVE_REGISTRY.md (the executed windows, their
pre-flight hashes and release read-backs).

**Cross-reference**: OL-27 (same-condition A/B — this entry qualifies its rule 1), OL-15 and PB-10 (shared-NPU contention, a different source of the same kind of error).
- **Evidence**: 内部路径（不公开）agent-team-wiki/projects/priority24/NPU_LIVE_REGISTRY.md (the eight-window factorial, the tile sweep with its accidental null arm, and the rotated re-run, each registered before its container was created and released with read-back).
- **Evidence (rule 7 applied, same fused-op family)**: 4-4 `4_GroupedMatmulSwigluQuant` is a different op variant of the same fused-op family as this entry's other evidence (4-4 V1 vs 4-11 `11_GroupedMatmulSwigluQuantV2` V2), on the same SoC (archive target Ascend950PR_9579 / CANN 9.1.0.B150, `output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/README.md:3`). This bullet shows rule 7 applied, not independently validated. One interleaved OLD/AB/ABC/OLD window over the frozen 60 cases on 2026-09-06, recorded in the kernel's tiling comment: `OLD 0.160250 / 0.162965 (same bytes twice => +1.69% window drift)` and `ABC vs AB: +0.71%, t=0.8, faster 35/60 -- SMALLER than the window's own drift.` (`output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/kernel/grouped_matmul_swiglu_quant_tiling.h:51-56`). `dbL0C=2` was not adopted: the comment reads it as buying "no measurable time" while filling all 256 KB of L0C, and the shipped value is `config.dbL0C = 1;` (`output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/kernel/grouped_matmul_swiglu_quant_tiling.h:57-62`). The comment does not name its timer; the archive's own F timing is `torch.npu.Event` device time (`output/npukernelbench/src/kernels/4_GroupedMatmulSwigluQuant/README.md:95`), and no msprof measurement is claimed. A wider multi-window survey exists at 内部路径（不公开）agent-team-wiki/projects/op4_4/FINDING_20260907T1835Z_DS_OP4_4_PAIRED_ARM_SURVEY_VS_NOISE_FLOOR.md; that page gives only summary figures, not its per-pair rows, so its distribution is not reproduced here.
