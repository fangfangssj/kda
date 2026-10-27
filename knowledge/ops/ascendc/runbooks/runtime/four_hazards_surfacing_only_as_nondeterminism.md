---
type: Runbook
title: "Four hazards that surface only as run-to-run non-determinism: early return above buffer-init, SetFlag<MTE2_V> before its producer DataCopy, missing SyncVtoS before scalar SetValue, loop-carried WAR needing SyncVtoMTE2"
description: "Four hazards that surface only as run-to-run non-determinism: early return above buffer-init, SetFlag<MTE2_V> before its producer DataCopy, missing SyncVtoS before scalar SetValue, loop-carried WAR needing SyncVtoMTE2"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-423"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-423 — OL-423: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=elementwise-attention-fused; scope=kernel-sync-correctness。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0; dtype=fp32; mode=AIV-only-SIMD。

**OL-423** — **Four hazards that surface only as run-to-run non-determinism: an early `return` inserted above a function's buffer-init block; `SetFlag<MTE2_V>` placed BEFORE the `DataCopy` that produces the data (the flag must FOLLOW the producer — contrast V→MTE3, where both Set and Wait correctly precede the store); a missing `SyncVtoS()` before a scalar `SetValue` to a vector-written buffer (WAW); and a loop-carried WAR in a combine loop needing `SyncVtoMTE2()`.** The WAR one is invisible at a degenerate single-tile config — the body runs once, so clearing that config is NOT evidence about the loop (qualifies OL-214). Detection heuristic: when a file has a correct sync helper AND open-coded sites, diff them (`grep -A1 'SetFlag<HardEvent::MTE2_V>' \

- Former A5Ops ID: `OL-423`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=elementwise-attention-fused; scope=kernel-sync-correctness`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0; dtype=fp32; mode=AIV-only-SIMD`
`unverified_on: other SoCs; fp16/bf16 paths; and this op's PRODUCT precision (canonical oracle non-regenerable) — only the 10-shape differential is verified`

**Anchor**: op57 ParallelPolarizedSelfAttention, 2026-08-28, Ascend950DT_9582 / CANN 9.1.0,
AIV-only SIMD. Symptom: same binary, two runs, two different answers (max-abs ~0.8 on outputs
of order 1). Four independent defects, each individually sufficient, all invisible to any
single run. Companion to OL-25 (TBuf has no automatic sync), OL-94 (TQue vs TBuf decision
table), OL-247 and CAND-A5OPS-0014 (PipeBarrier does not order a cross-pipe re-writer). What is new
here is (1) the SetFlag **placement** rule, (2) the early-return class, which is not a pipe
issue at all, and (3) why a degenerate tiling config cannot detect the loop-carried one.

### H1 — an early `return` that skips buffer initialisation
`Process()` zeroed five VECCALC accumulators with the comment "UB content is undefined at
Init; padded tails must be 0, never inf/nan". A later multi-phase dispatch
(`if (phase_ != 0) { ...; return; }`) was inserted **above** that block, so every phase path
ran with undefined padded tails. `Stage4` does `ReduceSum(uchw, red, cA_)` over the **aligned**
width and divides by the **logical** width to form a LayerNorm mean — the undefined tail lands
in the mean **before** the code that zeroes that tail runs.
**Rule**: initialise, then dispatch. When adding an early return to a function that already
had setup code, state which setup the new path skips. Grep the function for `Duplicate(`/
`memset`-like init and confirm every `return` is below it.

### H2 — `SetFlag<MTE2_V>` placed BEFORE the producing `DataCopy`
```cpp
SetFlag<HardEvent::MTE2_V>(id);   // WRONG: flag set before the producer is issued
DataCopy(ub, gm, n);
WaitFlag<HardEvent::MTE2_V>(id);  // already satisfied -> consumer races the in-flight DMA
Add(dst, ub, x, n);
```
For MTE2→V the **producer is the copy**, so the flag must follow it:
```cpp
DataCopy(ub, gm, n);
SetFlag<HardEvent::MTE2_V>(id);
WaitFlag<HardEvent::MTE2_V>(id);
```
Contrast V→MTE3, where the producer is the preceding vector op and both Set and Wait
correctly precede the store — which is exactly why the two read as symmetric and the wrong
one looks plausible.
**Detection heuristic**: the file already contained a correct helper (`SyncMTE2toV()` = Set
then Wait, called *after* the copy). When a codebase has both a helper and open-coded sites,
diff them — `grep -A1 'SetFlag<HardEvent::MTE2_V>' | grep DataCopy` returns non-zero only for
the broken ones.

**Where the 5 sites were, and why the archived control counts 2 (they are different
populations — do not read one as the other).** 5 = **2 inherited** (`Stage1Query`,
`Stage3RowReduce`) + **3 newly written** (`PhaseTwo` x2, `PhaseThree` x1). The 3 new ones
exist *because* the new code copied the inherited broken idiom — which is the transmissible
part of this lesson: an open-coded wrong barrier propagates by imitation every time the file
is extended, while the correct helper sitting beside it goes unused. The two archived
snapshots used as the control **predate the phase code**, so 2 is the complete count *for
those files*; it is not a count of the defect at the time of the fix.

### H3 — missing V→S before a scalar write to a vector-written buffer
`Subs(uchw, uchw, mean, cA_);` then `uchw.SetValue(i, 0.0f)` with no `SyncVtoS()` — cross-pipe
WAW. Same shape after `Add(zp, zp, vecB, c2A_)` → `zp.SetValue(i, -inf)`, and after an `Axpy`
loop → `q.SetValue(...)`. Three sites in one kernel; the surrounding code had `SyncVtoS()`
before the scalar *reads*, which made the omission before scalar *writes* easy to miss.

### H4 — loop-carried WAR in a combine loop (V→MTE2)
```cpp
for (t = 0; t < T; ++t) {
    DataCopy(acc,  ws[off  + t*cA], cA);   // clobbers acc while the PREVIOUS
    DataCopy(acc2, ws[off2 + t*cA], cA);   // iteration's Add may still read it
    SetFlag/WaitFlag<MTE2_V>;
    Add(xb, xb, acc, cA); Add(xb2, xb2, acc2, cA);
    SyncVtoMTE2();                         // <-- REQUIRED, was absent
}
```
**This is the one a degenerate config cannot catch.** With one tile the body executes once,
there is no previous iteration, and the kernel is perfectly deterministic. A "single-core
first" pass (OL-214) is still the right ladder, but record that it leaves loop-carried edges
untested — clearing the degenerate config is not evidence about the loop.

### Controls that localised these in ~4 builds
1. **A configuration that reduces to the known-good original.** Forcing the pre-patch code
   path at one tile gave two bit-identical runs whose per-case sums reproduced the recorded
   baseline exactly — clearing card, harness, weights and input binding in one step, so every
   later divergence was known to be new code.
2. **Degenerate vs real axis** (1 tile vs many) separates "machinery wrong" from "partitioning
   wrong". H1 showed at one tile; H4 needed many.
3. **Dump the intermediate GM buffer, not just the output.** The workspace was bit-identical
   across runs while the output differed, which proved the producing phases were sound and
   confined the fault to UB state in the consuming phase. That single measurement refuted
   every "the barrier is not working" hypothesis, all of which were wrong.

### Control for the H2 heuristic (run before trusting it)
`grep -A1 'SetFlag<HardEvent::MTE2_V>' <file> | grep -c DataCopy` on three snapshots of the
same kernel: **2** on the pre-multilaunch source, **2** on the SyncAll-era source (defect
present in both), **0** on the fixed source. Stated so the next reader knows what it prints
with the defect present AND absent, rather than only that it is cheap to run.

Read the count as *mis-ordered barriers present in that file*, and note two limits:
- The snapshots above predate the phase code, so their 2 is not the 5 fixed at head — see
  the site breakdown under H2.
- `grep -A1` inspects only the NEXT line, so a mis-ordered site with any line between the
  `SetFlag` and its `DataCopy` is missed. **Non-zero localises; zero is NOT proof of
  absence.** (The correct `SyncMTE2toV()` helper also matches the `SetFlag` pattern but has
  no `DataCopy` on the following line, so it is correctly not counted — which is why the
  fixed file has six `SetFlag<MTE2_V>` occurrences and a count of zero.)

### Boundary
Verified on Ascend950DT_9582 / CANN 9.1.0, AIV-only SIMD, fp32. Evidence is a 10-shape
differential against the pre-patch kernel with persisted fixed weights: after the fixes,
three runs 10/10 bit-identical and within 2.136e-07 (~1.8 fp32 ULP) of the single-block
reference. **This op's product precision was NOT verified** — its canonical oracle is
non-regenerable (`risk_class: process_hashed_hidden_state`), so no n/50 verdict backs this
entry. The hazards are code-level and mechanism-level; the precision claim is scoped to the
differential and must not be quoted as a product PASS.
