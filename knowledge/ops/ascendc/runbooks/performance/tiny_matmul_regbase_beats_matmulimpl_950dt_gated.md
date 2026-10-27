---
type: Runbook
title: "Tiny matmul (m,n≤16, k≤256) on 950DT: a hand-rolled regbase path beat MatmulImpl by +63% on gated cases — the win does not transfer across SoC"
description: "Tiny matmul (m,n≤16, k≤256) on 950DT: a hand-rolled regbase path beat MatmulImpl by +63% on gated cases — the win does not transfer across SoC"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-488"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-488 — OL-488: Tiny matmul (m,n≤16, k≤256) on 950DT: a hand-rolled regbase path beat MatmulImpl by +63% on gated cases — the win does not transfer across SoC

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul tiny-shape path (m,n≤16, k≤256 gate)。
- 验证来源：1_BatchMatmul LAYER-3 tiny gate (2026-09-08, Ascend950DT_9582, CANN 9.1.0; four-arm same-window table + non-author graft rebuild)。

**OL-488** — measured on 1_BatchMatmul (950DT_9582): for tiny shapes under the gate m,n≤16, k≤256, a hand-rolled single-block regbase mmad path beat the MatmulImpl library path by +63% (0.9116 → 1.4818) on the gated cases. The attributed cause — that the library's per-launch setup dominated at this size — is INFERRED from the arm structure, not isolated (see the mechanism note).

- Former A5Ops ID: `OL-488`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul tiny-shape path (m,n≤16, k≤256 gate)`
`verified_on: 1_BatchMatmul LAYER-3 tiny gate (2026-09-08, Ascend950DT_9582, CANN 9.1.0; four-arm same-window table + non-author graft rebuild)`

**Measured (the load-bearing facts)**: four-arm same-window table (Ascend950DT host card6, profiler device-task domain): gated tiny(24) cases 0.9116 → 1.4818 with the gate; reproduced across two unrelated lineages (db0 split vs main split) and by a non-author graft arm (`019a0b61`); ALL(51) 0.9530 → 1.1330/1.1332; P51 51/51 both gate arms (`l3_tinygate_20260908T2030Z/p51/p51.l3_kimi.json`, `…/p51/p51.graft_mainhead_tinygate.json`, and the non-author rerun `…/nonauthor_review/p51/p51.nonauthor_cand.json` — all in-repo). The gate's cost side (untouched cases 1.0007→0.8999) is the triton-lane gate-codegen-cost entry (not yet on main).

**Boundary vs P-P119 (A3)**: P-P119 (`patterns/domains/a3_mix_small_matmul_cube.md`) is the A3 (Ascend910_9382/arch22) MIX many-small rule — hundreds of tiny matmuls per op inside chunked recurrence, discriminator msprof `aic_scalar ≫ aic_mac`. This instance is a SINGLE tiny matmul per launch on 950DT where the measured win is consistent with the library per-launch setup dominating (cause INFERRED, not isolated); same principle family (library setup dominating tiny matmul), different SoC/arch, call pattern, and discriminator. Neither entry's measurements support the other's SoC.

**Mechanism note (INFERRED cause)**: the library-setup-dominance attribution is inferred from what the arms changed (hand-rolled path removes the library call sequence) and is consistent with the ~2µs/launch library setup estimated during the L3 arm; no instruction-level isolation of "library setup vs other structure differences" exists. **Emulator boundary (not hardware generality)**: tikicpulib reported this regbase tiny-bmm path UNSUPPORTED on 3510 (emulator error) — the Fixpipe-only output stage is a workaround for THAT path under THAT emulator limitation, validated on-card; it is not a claim that 950 hardware requires Fixpipe-only output staging in general, and it is not a must for other implementations.

**Threshold provenance (bound to the measured audit)**: gate bounds = probe coverage + L1 buffer sizing (aBuf 16×256, bBuf 256×16), not case-set fitting — k gate 256 vs max observed k=32 (8× headroom; a fitted threshold would hug 32); m,n=16 sit at the case grid's coarse edge {…,16,32,…} so any bound in [16,31] partitions the 51 cases identically — the case set cannot resolve inside that interval, and widening needs a hardware probe.

**Evidence**: `output/npukernelbench/src/kernels/1_BatchMatmul/l3_tinygate_20260908T2030Z/` (README §4/§5 threshold audit, p51/, f_arms/, nonauthor_review/); `candidate/kernel/bmm_l3_regbase_kernel.h` on main; !909 chain (main `50b56e63a` + follow-ups) + !921 (`47df5f684`). Cross-ref [P-P119](../../examples/a3_mix_small_matmul_cube.md); the gate cost side lives in triton's lane (unmerged as of this writing).

**Other instances (predicted)**: tiny-shape ops where vendor per-launch setup is comparable to compute (tiny conv, small-N reductions) — measure the setup-vs-compute split before tuning compute.

---
