---
type: Runbook
title: "A performance plateau declared on an unmeasured assertion is not a plateau — occupancy must be measured before a ceiling is recorded"
description: "A performance plateau declared on an unmeasured assertion is not a plateau — occupancy must be measured before a ceiling is recorded"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-317"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-317 — OL-317: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (scalar-pipe-bound multi-core kernels)。
- 验证来源：soc=Ascend950PR; cann=9.0.0。

**OL-317** — **A performance plateau declared on an UNMEASURED assertion is not a plateau; occupancy must be MEASURED before a ceiling is recorded.** A scalar-pipe ratio near 1.0 is true per ACTIVE core and says nothing about how many cores are active — a saturated machine and a half-idle one report the same ratio. Discriminator: report `cube_util` before/after. Measured: an unmeasured 'smaller tile ⇒ worse scalar' assertion closed the search; measuring occupancy recovered **−21.8% median device time** (idle cores 8–16/24 → 22–24/24, `cube_util` 53% → 71.9%). True plateau is AFTER the occupancy lever. Corollary: a tool that fails only on a `Health=Alarm` device is indistinguishable from a broken tool. backend=ascendc.

- Former A5Ops ID: `OL-317`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (scalar-pipe-bound multi-core kernels)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9392 (arch 220x — core counts and scheduling differ; re-measure before reusing the numbers)`
- **Category**: performance_methodology
- **Loaded by**: Optimizer
**Principle**: an optimizer may only record a terminal performance verdict after the *occupancy*
lever has been **measured**. Reasoning about it is not measuring it. A ceiling attributed to a
saturated pipe, while cores sit idle, is a statement about the search, not about the hardware.
**Why it is easy to get wrong**: a scalar-pipe ratio near 1.0 reads as "the pipe is the bottleneck,
nothing else matters". That is true *per active core* and says nothing about how many cores are
active. Both a genuinely saturated machine and a half-idle one report a high scalar ratio.
**Discriminator — use the one YOUR unit actually reports (report it before/after either way)**: a
saturated scalar ratio has THREE readings and the ratio alone cannot separate them.

| unit | occupancy measurement | reading |
|---|---|---|
| cube / AIC | `cube_util` | high scalar + low `cube_util` = unfilled machine; high scalar + high `cube_util` = real ceiling |
| pure-vector / AIV | `Block Num` in `op_summary_*.csv`, against the chip's physical AIV count | `Block Num` equal to a problem dimension (rows, heads, groups) means occupancy is bounded by that dimension, not by the hardware |

⚠️ `cube_util` **does not exist for a `KERNEL_TYPE_AIV_ONLY` kernel**, so an AIV-only optimizer
reading this entry for `cube_util` alone concludes the discriminator is unavailable and skips the
check — which is the exact failure this entry exists to prevent. `Block Num` is the AIV-side form
and it is in the same CSV.

**Third reading — fixed per-launch cost.** If the *most* pipe-saturated case is the one you already
WIN, the ratio is inverted with respect to work, and a large part of that pipe's cost is fixed per
launch rather than proportional to it. Separate the two terms by fitting measured cost across two
shapes that differ in work: `cost ≈ per_unit × units + fixed`. A fixed term that dwarfs the
per-unit arithmetic is launch overhead (queue bookkeeping, per-iteration descriptor construction,
flag triplets, scalar tiling reads) and no amount of tuning the *inner loop* will touch it — the
levers are fewer launches, cheaper launch setup (OL-77 scalar kernel args), or moving the cost off
the launch path entirely (CAND-A5OPS-0007 host-side caching).
**Evidence — pure-AIV instance** — `SpatialGroupEnhance`, Ascend950PR_9579, CANN 9.1.0.B060
(2026-08-21): `aiv_scalar_ratio` dominated every case (0.770 / 0.809 / 0.871 on cases 32 / 26 / 17)
against `aiv_vec_ratio` 0.43 / 0.32 / 0.17 — and case 17, the MOST scalar-saturated of the three,
was the case the kernel already won, at 1.11×. That inversion is the third-reading tell above.
Measured rather than asserted: `Block Num` == R (rows), so the worst case (R=2) ran on **2 of 48
AIVs** while each of those two cores walked 256 channels serially; per-case speedup tracks C, not
tensor size (mean by C: C=2 → 0.70, C=16 → 0.50, C=64 → 0.36, C=256 → 0.18). The two-term fit over
cases 26→32 (32 → 512 channel iterations) gave ≈0.217 µs per channel-iteration + ≈5.7 µs fixed per
launch, against only ~160–400 fp32 lanes of actual arithmetic per iteration ⇒ the per-iteration
cost is overhead, not compute. **No ceiling was recorded**: both remaining levers are named and
unexploited (cross-core channel split with an OL-252 fixed-order reduction to keep
determinism-by-construction; OL-77 scalar-arg marshalling against the fixed ≈5.7 µs).

**Evidence** — `flash_attention_score_causal_grad`, Ascend950PR, CANN 9.0.0 (2026-06):
- An optimizer pass asserted "smaller inner-block ⇒ worse scalar ratio" and, on that basis, treated
  the tile dimension as exhausted. The assertion was **never measured**; the one tile change it did
  try moved nothing (176us → 176us, Δ0%) and was reverted.
- A later pass measured occupancy and applied an adaptive ceil-fill (fill up to MAX_CORES, no
  overshoot): **device-time median −21.8%** across 11/12 cases, with determinism and precision floors
  held. Idle cores went from 8–16 of 24 to 22–24 of 24.
- Re-confirmation pass: Δ0% with occupancy genuinely exhausted, `aic_scalar` 0.955 and `cube_util`
  **71.9%, up from 53%** — the rise is what shows the earlier number was an unfilled machine.
- ⇒ **The true plateau is after the occupancy lever, not before it.** The residual per-GEMM scalar
  ceiling is architectural (the fused rewrite was separately ruled out), so further gains require a
  GEMM-count change, which is outside optimizer scope.
**Other instances (predicted)**: any multi-core kernel whose per-core bottleneck metric saturates
before the core count does — reductions with small tail blocks, attention variants with
per-head dispatch, grouped ops whose group count is below the core count.
**Corollary — instrument health before instrument blame**: in the same sequence a "profiler parser
outage" was reported and re-attributed on re-measurement: the parser was fine, two devices were in
`Health=Alarm` (aicore timeout `507014`). ⇒ A tool that fails only on a sick device is
indistinguishable from a broken tool. Re-run on a known-healthy device before recording a tooling
- **Cross-reference**: OL-43 (single-block test as the multi-core contention diagnostic — the
  complementary direction: OL-43 isolates contention, this entry catches under-fill); OL-200
  (correctness does not imply perf-normality on a MIX unit — a half-idle schedule is correct);
  OL-214 (single-core-first testing).
