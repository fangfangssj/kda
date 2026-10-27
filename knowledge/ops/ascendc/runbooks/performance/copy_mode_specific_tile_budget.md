---
type: Runbook
title: "Copy engines with multiple copy modes have opposite tile-size optima per mode — one TILE_BYTES serves neither"
description: "Copy engines with multiple copy modes have opposite tile-size optima per mode — one TILE_BYTES serves neither"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-COPY-MODE-SPECIFIC-TILE-BUDGET"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-COPY-MODE-SPECIFIC-TILE-BUDGET — Copy engines with multiple copy modes have opposite tile-size optima per mode — one TILE_BYTES serves neither

## 适用场景

- 适用平台与范围：soc=Ascend950PR (351x family)；CANN 9.1.0；含 >=2 种并行结构不同 copy 模式的纯拷贝/数据搬运 kernel。
- 验证来源：soc=Ascend950DT_9582、CANN 9.1.0（13_Cat，2026-09-16，A/B 实测）；soc=Ascend910_V220 (A3) 与 dav-m200 未验证。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**CAND-COPY-MODE-SPECIFIC-TILE-BUDGET** — a cat/copy engine whose modes differ in how tile count maps to core parallelism needs a per-mode tile budget, not a single `TILE_BYTES`. A line-batched mode (many small lines; tile count bounded by outer-lines/k) optimizes for tile COUNT: small-line shapes get few tiles → few of the 64 cores busy; halving the per-tile budget doubles tiles → doubles cores → strided-write time halves. A within-line mode (parallelism = bytes/tile, already ≥ cores on mid shapes) optimizes for per-tile BYTES: its per-tile serial MTE2→MTE3 stall DOMINATES at 1–6 tiles/core, so smaller tiles multiply the serial stalls — a REGRESSION. Give the count-bound mode its own smaller budget (e.g. LB_TILE_BYTES=48KB) while the bytes-bound mode keeps the large one (96KB/depth-2).

`applies_to: soc=Ascend950PR (351x family); cann=9.1.0; bisheng=n/a; op_class=pure-copy / data-movement kernels with ≥2 copy modes of different parallelism structure`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 (13_Cat, 2026-09-16)`
`unverified_on: soc=Ascend910_V220 (A3); dav-m200`

- **Concrete anchor**: `if (mode == LINE_BATCHED) tileBytes = 48*1024; else tileBytes = 96*1024;` — same kernel, mode-keyed budget. Mechanism check via msprof: small-line case (112B lines) went 10→20+ tiles on 64 cores, kernel 8.01→6.10µs, mte3 4.62→2.45µs.
- **Evidence (A/B, KEEP)**: 13_Cat ko-2 Opt3 (2026-09-16): line-batched-only 48KB budget → whole-suite perf 0.9393→0.953×; small-line cases +0.11..+0.15; within-line cases flat as designed. Precision re-verified 51/51 OVERALL_T1 + det 51/51 on the Opt3 kernel.
- **Evidence (REVERT — single budget for both modes)**: 13_Cat ko-2 Opt2: depth-4 + 48KB everywhere → within-line mid-size cases −0.10..−0.16 (smaller tiles doubled per-tile serial stalls); net wash, reverted. (REVERT: hypothesis "one smaller budget helps all modes" invalidated.)
- **Ruled out (measured, same spawn)**: dual-queue software pipeline splitting issue/drain — correct-by-construction, 0 gain (per-tile MTE2→MTE3 serialization invariant to queue structure); host-side plan hoisting — perf gate measures device time only, host work invisible.
- **Other instances (predicted)**: split/gather/scatter/segment-copy engines with a many-small-chunks mode alongside a large-contiguous mode; any kernel where one mode's parallelism is count-bound and another's is bytes-bound.
- **Cross-ref**: OL-63 (tile-first, queue-depth-second for elementwise-vec — this candidate is the copy-engine analogue; a depth-4-with-smaller-tiles nuance for copy engines is parked as a separate follow-up), OL-42 (the residual gap is mechanism-specific knowledge, not hardware), OL-85 (mode/geometry-class-based gating, not test-value-specific), CAND-QUEUE-PROTOCOL-IS-THE-MTE2-MTE3-ORDERING (the serial-stall mechanism's ordering substrate). backend=ascendc.
