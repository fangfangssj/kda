---
type: Runbook
title: "A block-split gate for batched/per-sample ops must key on per-batch starvation, never on total problem size"
description: "A block-split gate for batched/per-sample ops must key on per-batch starvation, never on total problem size"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-482"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-482 — OL-482: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579/Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=batched matmul and any per-sample op considering multi-block fan-out。
- 验证来源：1_BatchMatmul gate-v3/v4 full-denominator A/B (2026-09-07, 51 cases, alternating same-session fleet evals)。

**OL-482** — a block-split gate for batched/per-sample ops must key on PER-BATCH starvation (per-batch work vs one core's throughput), never on total problem size. Shapes with batch>1 already own native block parallelism (blockDim=batch); splitting them further only multiplies the per-block MatmulImpl setup cost (~1.7us/block measured) across tinier tiles. Measured on 1_BatchMatmul: total-MAC>=1M gate regressed batch>1 fp16/bf16 `[b,64,128]` cases by 0.09-0.17 profiler ratio while the batch=1 `[1,128,256]x[1,256,128]` class gained 0.306->0.977; per-batch-MAC>=2M gate keeps all wins with zero regressions. Fan-out knee at 16 blocks for this class (1/2/4/8/16/32 = 13.54/8.34/5.76/4.61/4.60/4.94us; vendor 4.18us).

- Former A5Ops ID: `OL-482`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR_9579/Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=batched matmul and any per-sample op considering multi-block fan-out`
`verified_on: 1_BatchMatmul gate-v3/v4 full-denominator A/B (2026-09-07, 51 cases, alternating same-session fleet evals)`

**Principle**: parallelize only where starvation exists. A gate on total FLOPs/elements fires on shapes that are already parallel across the batch dimension, where extra blocks shrink tiles until fixed per-block costs dominate; a gate on per-sample work fires exactly where one core is stranded with a big problem. Sweep the fan-out before picking a default — the knee is where per-block setup starts to outgrow the saved compute, and it is not at the vendor's block count.

**Concrete anchor**: 1_BatchMatmul `[8,64,128]x[8,128,64]` (4.19M total MAC): total-MAC gate splits to 32 blocks of 16x64 tiles → profiler ratio 1.06→0.89 (fp16), 1.10→0.94 (bf16). Same total MAC at batch=1 (`[1,128,256]x[1,256,128]`): 0.31→0.89-0.98 in all three dtypes. Per-batch MAC separates the two exactly (0.52M vs 4.19M; gate at 2M).

**Evidence**: per-case attribution table in SWEEP_case4_20260907.md + f51_{tile16,noorg,db0_r2,tilev4}.json receipts in the 1_BatchMatmul delivery closure (candidate_history/snapshots/5e701e77…/arms-rejected/ and evidence/).

**Other instances (predicted)**: any op with a batch or group dimension and a multi-core fan-out temptation (GroupedMatmul, MoE experts, batched attention); the same per-slice starvation key applies, and coverage-safe tile sizing (`base = align(ceil(dim/tiles))`) is required or edge shapes silently drop columns (latent v3 bug: fixed baseN=32 with nTiles=1 left columns 32..n uncomputed).
