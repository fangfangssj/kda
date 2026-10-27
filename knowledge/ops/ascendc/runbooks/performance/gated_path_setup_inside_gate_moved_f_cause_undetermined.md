---
type: Runbook
title: "Restructuring Init so a gated path's setup runs only inside the gate moved the measured F on 1_BatchMatmul — the cause is undetermined"
description: "Restructuring Init so a gated path's setup runs only inside the gate moved the measured F on 1_BatchMatmul — the cause is undetermined"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-486"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-486 — OL-486: Restructuring Init so a gated path's setup runs only inside the gate moved the measured F on 1_BatchMatmul — the cause is undetermined

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul-class µs-scale kernels with a shape-gated fast path。
- 验证来源：1_BatchMatmul Init-slim arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, same-window fleet-consumer A/B, two runs per arm)。

**OL-486** — measured on 1_BatchMatmul (950DT_9582): restructuring the kernel Init so the tile-split gate's setup (block→tile div/mod, member init, branch setup) runs only inside the gated branch moved the measured device-task-time F on the fixed-51. What moved is certain; WHY (bookkeeping removal vs codegen/layout side effects of the restructure) is undetermined — see the mechanism section before citing a cause.

- Former A5Ops ID: `OL-486`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=1_BatchMatmul-class µs-scale kernels with a shape-gated fast path`
`verified_on: 1_BatchMatmul Init-slim arm (2026-09-07, Ascend950DT_9582, CANN 9.1.0, same-window fleet-consumer A/B, two runs per arm)`

**Measured (the load-bearing facts, each bound to its in-repo receipt)**: pre-slim fix-tree profiler GM 0.856811393107 / 0.857922700550 (`f51_fixv2_r1.json` blob `20ad0253233f`, `f51_fixv2_r2.json` blob `8848dbde7875`) → slim-tree profiler GM 0.939961 / 0.940140 (`f51_slim_r1.json` blob `e837065706d7`, `f51_slim_r2.json` blob `1ece7b4b0883`); per-case ≥1.0 count 4 → 24 in both run pairs; P51 51/51 bit-exact unchanged (`fixed51_grade.slim.json` blob `7c4d6256c9fc`); shipped kernel.h = the measured bytes, sha256 `162c815ebf98abe1ef0711e5ddffa984b6a87d894785d3a7f7b9877f2a716425` at `62389e6ce:output/npukernelbench/src/kernels/1_BatchMatmul/candidate/kernel/batchmatmul_kernel.h` (!888 tree — NOT current main, which later gained the tiny-gate graft). The gate that owns the restructured setup fires on 3/51 cases (fixed51 idx 4/9/14, per the takeover partition, `l3_tinygate_20260908T2030Z/README.md` §4). The measured time domain is the KERNEL's device-task time; this entry claims nothing about host-side execution.

**Mechanism (undetermined — do not read as isolated causality)**: the restructure moved a BUNDLE at once (the setup code now runs only inside the gate, plus whatever codegen/layout side effects the new structure produced), and the measured delta belongs to that bundle. The retained evidence does not isolate bookkeeping work from codegen/layout effects, and there is no instruction-level attribution; no cause is claimed. What is load-bearing is the measured restructure and its receipt-bound numbers above; why they moved remains hypothetical. A reader needing the cause must isolate it (e.g. re-add only the setup behind the gate vs only the structural change) before attributing it.

**Evidence (repo-resolvable)**: fix-tree receipts `…/candidate_history/snapshots/5e701e77…/arms-rejected/f51_fixv2_r1.json` (blob `20ad0253233f`) and `…/f51_fixv2_r2.json` (blob `8848dbde7875`); slim receipts `…/evidence/f51_slim_r1.json` (blob `e837065706d7`) and `…/evidence/f51_slim_r2.json` (blob `1ece7b4b0883`); grade `…/evidence/fixed51_grade.slim.json` (blob `7c4d6256c9fc`); shipped source `62389e6ce:…/candidate/kernel/batchmatmul_kernel.h` (sha256 `162c815ebf98…`); merges !872/`cb8f01c05` and !888/`62389e6ce`; takeover partition table in `…/l3_tinygate_20260908T2030Z/README.md` §4 — all on main.

**Other instances (predicted)**: hot µs-scale kernels whose Init grew feature-by-feature — profile the Init path when small-case device time drifts above the minimal-baseline path.
