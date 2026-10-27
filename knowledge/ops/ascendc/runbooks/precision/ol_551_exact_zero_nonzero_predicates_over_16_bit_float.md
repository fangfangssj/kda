---
type: Runbook
title: "exact zero/nonzero predicates over 16-bit float tiles on narrow-VEC SoCs (dav-m200)"
description: "16 位浮点的精确零/非零谓词：纯线性半字提取数学上不可行（有证明）；正确做法是 int16 位承载 + 标量判 (bits & 0x7FFF) != 0"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-551"]
---

# exact zero/nonzero predicates over 16-bit float tiles on narrow-VEC SoCs (dav-m200)

**OL-551** — **exact zero/nonzero predicates over 16-bit float tiles on narrow-VEC SoCs (dav-m200)**: (1) linear-only half-word extraction is PROVABLY impossible (`+,-,<<` mod 2^32 all collapse to `k·x`; killing the high half forces `k≡0 mod 2^16` which kills the low half — `2·(b32−(b32<<16))` tests pair-ANY-nonzero, false-neg on odd=−0.0); (2) scalar reads of V-pipe results need an explicit V→S fence BEFORE the read (stale signature: `count==pairs` all-ones); (3) the exact predicate is scalar `(bits & 0x7FFF) != 0` on int16-carried tiles — ±0 excluded, subnormal/NaN/±Inf kept. Decision rule: don't build exact 16-bit predicates from VEC half-word algebra — carry int16, test magnitude bits scalar-side. 22_Nonzero 310p 2026-08-15: 160/160 + 50/50 + 518/518 bit-exact post-rewrite. A5 P-P80 NOT invalidated (arch-scoped: native 32-lane VEC). Cross-ref [`OL-195`](../compilation/ol_195_v220_v351_vec_compute_chain_port_pipebarrier_pipe.md), [`OL-25`](../compilation/ol_25_tbuf_has_no_automatic_synchronization_must_use_tqu.md), CAND-DAV-M200-VEC-NAN-EMISSION (2nd witness), CAND-DAV-M200-GATHER-SELECT, [`PB-120`](../compilation/pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md), P-P80. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, arch-200, 16-lane VEC); cann=9.0.0; bisheng=n/a (the algebra sub-rule is compiler-independent); op_class=predicate-driven stream compaction (nonzero / where / masked_select / sparse-COO build); dtype=fp16/bf16`
`verified_on: soc=Ascend310P3; cann=9.0.0 (22_Nonzero 310p, 2026-08-15: 160/160 probe sweep + 50/50 Pass A + 518/518 Pass B bit-exact after the scalar-predicate rewrite)`
`unverified_on: soc=Ascend950PR (arch35 has native 32-lane VEC + working vcmp NaN — the A5 CompareScalar+GatherMask design P-P80 remains canonical there and is NOT invalidated by this entry)`

### Principle — three traps that co-occur when predicating 16-bit floats through the VEC unit on a narrow-VEC SoC

1. **Algebra (provable, compiler-independent)**: with only linear ops (`+`, `-`, `<<`, all mod 2^32) a 16-bit half of a packed 32-bit pair CANNOT be isolated — every composition collapses to `x ↦ k·x mod 2^32`; zeroing the high half for ALL inputs forces `k ≡ 0 (mod 2^16)`, which also zeroes the low half. The tempting "pair-domain" trick `oddT = 2·(b32 − (b32<<16))` therefore tests "**the pair has ANY nonzero element**", not "the odd half is nonzero": false-positive for even≠0/odd=0, false-NEGATIVE for odd=−0.0 (`2·0x80000000 ≡ 0 mod 2^32`).
2. **Fence**: a scalar read of V-pipe results (packed compares, masks) needs an explicit **V→S fence BEFORE the read** (the OL-195 / OL-25 class). Measured stale signature: merged popcount `count == pairs` EXACTLY — the all-ones uninitialized read.
3. **Exact predicate**: on int16-carried 16-bit float tiles, `(bits & 0x7FFF) != 0` is exact for **every** IEEE-16 encoding — ±0 excluded (both sign variants), subnormal / NaN / ±Inf kept — and it runs scalar, immune to the 16-lane VEC predicate landmines (CAND-DAV-M200-VEC-NAN-EMISSION, PB-120).

**Decision rule**: when a zero/nonzero (or presence) predicate over fp16/bf16 data must be EXACT on a narrow-VEC SoC, do not build it from VEC half-word algebra at all — carry the tile as int16_t and test the magnitude bits scalar-side.

### Evidence
- 22_Nonzero 310p kw-1 (2026-08-15, benchmark): the VEC pair-domain path failed BOTH axes — algebra (61/80 sampled mismatches) and fence (count==pairs uninit signature); the fix removed 16-lane VEC predication entirely and unified fp16/bf16 with the int paths under the scalar predicate → 160/160 probe sweep + 50/50 Pass A + 518/518 Pass B bit-exact (`22_Nonzero/failures_ledger.md` D-1).

### Other instances (predicted)
- where / masked_select / sparse-COO build / scatter-with-mask on dav-m200 — same exact-predicate requirement over 16-bit floats.
- Any int16-carried 16-bit data movement (CAND-DAV-M200-INT16-BF16-DATAMOVE) that later needs a per-element predicate — the carrier and the predicate compose directly.

### Cross-references
OL-195 (V→S fence class), OL-25 (no-auto-sync), CAND-DAV-M200-VEC-NAN-EMISSION (2nd witness of the 16-bit-VEC-predicate trap class — see its 22_Nonzero row), CAND-DAV-M200-GATHER-SELECT (VEC select substitute when a SELECT rather than a zero-test is needed), PB-120 (16-bit vconv cast hazards), P-P80 (A5 canonical predicate design — arch-scoped, not invalidated). backend=ascendc.
