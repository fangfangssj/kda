---
type: Runbook
title: "Read the per-kernel pipe shares before choosing an AIV optimisation — a correct mechanism argument can still point at a minor pipe"
description: "Read the per-kernel pipe shares before choosing an AIV optimisation — a correct mechanism argument can still point at a minor pipe"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-442"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-442 — OL-442: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC AIV kernel optimisation selection; bisheng=n/a; op_class=im2col/col2im scatter-gather。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 (3-10 ConvTranspose2d, 2026-09-06)。

**OL-442** — **read the per-kernel PIPE SHARES before choosing an AIV optimisation; a correct mechanism argument can still point at a minor pipe**. On 3-10 the loop structure argued for collapsing many short vector `Add`s into few long ones — but on that kernel **`aiv_scalar_ratio` (0.694) is far higher than `aiv_vec_ratio` (0.163)**, so the argument targeted a pipe carrying much less of the time. (It is NOT the smallest share — `aiv_mte3_ratio` is 0.025.) Rank candidates on duration, occupancy AND source-code shape together; shares alone do not decide, and do not arithmetise them into recoverable-time budgets. See OL-317: a pipe ratio is per ACTIVE core and says nothing about occupancy.

- Former A5Ops ID: `OL-442`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; scope=AscendC AIV kernel optimisation selection; bisheng=n/a; op_class=im2col/col2im scatter-gather`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (3-10 ConvTranspose2d, 2026-09-06)`

**Problem**: a lever identified by *reading the loop* can be mechanically correct and still be
the wrong thing to build. In 3-10's `col2im` the destination rows are strided by `sh*sw*Mwp`
while the source band is contiguous, so the kernel issues `kh*kw*nrows` short `Add`s of length
`Mw`; a phase-major layout would collapse these into `kh*kw` full-length `Add`s, which is a
smaller instruction count for the same elements. That shape argument is sound on its own terms.
It is still not, by itself, a reason to build this first.

**Measurement (raw, unweighted per-kernel means).** Source: one retained perf bundle,
sha256 `7c0b3ecadca9d51c770619ab356ba5c60f878854d3cbdbca8697777a988eb30c`. Members selected are
the CANDIDATE-side `**/kernel_details.csv` entries only (bundle members whose path carries the
candidate marker; reference-side members excluded), and within them only rows whose `Name`
contains `convtranspose2d`. Denominator: **250 records per kernel** (50 cases x 5 repeats).

| kernel | `aiv_scalar_ratio` | `aiv_mte2_ratio` | `aiv_vec_ratio` | `aiv_mte3_ratio` |
|---|---:|---:|---:|---:|
| `col2im` | 0.694 | 0.195 | 0.163 | 0.025 |
| `pack` | 0.631 | 0.202 | — | 0.253 |

**Statistical / weighting convention — read before using these numbers.** These are *unweighted
means of per-record ratios*, not time-weighted shares. Multiplying such a mean by a summed kernel
time mixes a mean-of-ratios with a sum-of-times and is indicative only. The pipes also OVERLAP,
so pipe times are not additive with each other or with end-to-end kernel time.

**What the measurement supports**: an ORDERING. On both AIV kernels the scalar pipe carries the
largest share, well above vector. That was sufficient to deprioritise the vector-issue lever.

**What it does NOT support** (all three were written and withdrawn during this work):
1. that the scalar share is *all* address arithmetic;
2. that the scalar share converts one-for-one into recoverable microseconds;
3. that the vector share upper-bounds the end-to-end gain of a restructure — a restructure also
   changes MTE traffic and scalar work, and overlapping pipes make a share a poor bound.

**Pending interpretation, explicitly untested**: on kernels of this shape a substantial part of
the scalar pipe is plausibly per-row/per-tap address derivation (loop-carried bound computation,
per-column phase/remainder arithmetic, per-row parity filtering), and precomputed per-phase
tap/index tables would reduce it. This is a hypothesis for the NEXT experiment, not a result.
Its falsifier: if a prototype lowers the scalar ratio materially but total kernel time does not
fall with it, **the expected net gain is refuted**. That outcome does NOT on its own establish
WHY — overlap with other pipes is only one candidate explanation, alongside changed occupancy
(OL-317), shifted work into another pipe, or a cost the ratio never described. Diagnosing the
cause needs its own measurement.

**Technique to carry**: when a candidate optimisation is identified by reading code, pull the
per-kernel pipe ratios out of profiling you ALREADY HAVE before writing anything — on this op it
cost one script against a retained bundle, no device and no new run. Reading the loop tells you
what is happening; the shares add one more input to the ranking. **Rank on duration, occupancy
and source-code shape together** — a share alone neither condemns nor endorses a candidate, and
a low-share candidate can still be correct to build if it is cheap, unblocks another change, or
removes work from a busier pipe as a side effect. Cross-check occupancy per OL-317 before
treating any ratio as a ceiling.

### Evidence locators

- pipe ratios: `phase_o5_perf_raw_1788666483227259310.tar` -> `**/kernel_details.csv`
  (columns `aiv_scalar_ratio`, `aiv_mte2_ratio`, `aiv_vec_ratio`, `aiv_mte3_ratio`)
- loop under discussion: `kernel/convtranspose2d_kernel.h`, `ct2d_col2im_one_impl` inner
  `for q in [0, nrows)` band-accumulate loop
