---
type: Runbook
title: "Shape-adaptive `nblk` reduction for tiny workloads is NOT a guaranteed perf win — launch-path cost may dominate over idle-core cost"
description: "Shape-adaptive `nblk` reduction for tiny workloads is NOT a guaranteed perf win — launch-path cost may dominate over idle-core cost"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PERF-NBLK-LAUNCH-DOM"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PERF-NBLK-LAUNCH-DOM — Shape-adaptive `nblk` reduction for tiny workloads is NOT a guaranteed perf win — launch-path cost may dominate over idle-core cost

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=any kernel with small per-shape row/element counts (num_rows ≤ ~8, total work < one wavefront)（卡内 applies_to 行）。
- 验证来源：soc=Ascend950PR; cann=9.0.0 (ada_layer_norm only — single op evidence)（卡内 verified_on 行）；未验证范围：soc=Ascend910_V220 (A3 family — launch-path cost profile may differ); other op classes（卡内 unverified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=any kernel with small per-shape row/element counts (num_rows ≤ ~8, total work < one wavefront)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (ada_layer_norm only — single op evidence)`
`unverified_on: soc=Ascend910_V220 (A3 family — launch-path cost profile may differ); other op classes`

**Hypothesis**: Setting `nblk = min(56, num_rows)` for small workloads (so each AIC gets ≥ 1 row of real work) saves the idle-core scheduling cost on tiny shapes and should improve ratio vs vendor.

**Counter-evidence (ada_layer_norm kw-3 → kw-3-revert, 2026-05-13)**: Applying this rule to `(B=1, S=1, H=16)` and similar tiny cases dropped the perf ratio from 0.44× to 0.40× across the whole sweep. Root cause hypothesis: with fewer cores active, per-core work scales proportionally (1 core does what 56 cores would split), and that serialization cost exceeds the saved idle-core spin. More importantly, the **dominant cost on small shapes appears to be the CANN op-API launch path itself**, not core spinning — reducing core count cannot help if the bottleneck is upstream of the kernel.

**Recommended workflow**:
1. Before applying shape-adaptive `nblk` reduction, run `msprof` on the smallest-shape case and check whether AIV idle time is actually a meaningful chunk of the kernel's wall-clock. If launch-path is dominant, `nblk` tuning is a no-op (or regression) by construction.
2. If profile shows idle-core cost IS significant, prefer keeping `nblk=56` and accept the idle-core waste — the alternative often regresses.
3. If launch-path is the dominator, the real optimization is upstream (kernel-launch fusion, batched dispatch, persistent kernels) — not per-shape `nblk` knob twiddling.

**Promote-to-pattern criteria**: validated on ≥ 2 ops in different op_classes with `msprof`-confirmed launch-path-dominant profiles, plus an explicit measurement showing the `nblk` knob has zero or negative effect on those shapes.

**Related**:
- OL-127 (no single-thread SIMT — the upper bound `nblk` is still 56 / hardware max; this candidate is about avoiding aggressive reduction, not about the floor)
- P-P1 (numBlocks dynamic — the standard "always use all cores" pattern; this candidate documents a regime where deviating from it doesn't help)
- MSPROF_AGENT_GUIDE.md (the profile-first-then-tune workflow this candidate enforces)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PERF-NBLK-LAUNCH-DOM，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
