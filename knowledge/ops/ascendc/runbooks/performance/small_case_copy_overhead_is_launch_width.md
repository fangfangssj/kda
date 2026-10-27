---
type: Runbook
title: "Small-case cat/copy overhead on A5 is launch-WIDTH, not pipeline ceremony"
description: "Small-case cat/copy overhead on A5 is launch-WIDTH, not pipeline ceremony"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-SMALL-CASE-COPY-OVERHEAD-IS-LAUNCH-WIDTH"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-SMALL-CASE-COPY-OVERHEAD-IS-LAUNCH-WIDTH — Small-case cat/copy overhead on A5 is launch-WIDTH, not pipeline ceremony

## 适用场景

- 适用平台与范围：soc=Ascend950 (351x)；CANN 9.1.0；含微小 workload case 的纯拷贝/数据搬运算子。
- 验证来源：13_Cat M-I run-1 ko-2 Opt2（host 侧 block_dim=min(64, work_units) 实测 +81%）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950 (351x); cann=9.1.0; op_class=pure-copy / data-movement with tiny-work cases`
`verified_on: 13_Cat M-I run-1 ko-2 Opt2 (+81% via host-side block_dim=min(64, work_units))`

Fixed ~8.9us device time (kernel_details Duration) for a 64-AIV-core launch of near-zero work vs CANN ~2us. The V-bridge/pipeline-ceremony hypothesis is FALSIFIED (V-bridge removal measured neutral).
**Lever**: `block_dim = min(64, work_units)` host-side. **Falsifier for future 'pipeline ceremony' theories**: check kernel_details attribution on a near-zero-work case FIRST — if tiny-case Duration == large fixed constant, it's dispatch width, not per-tile pipeline.

- **Cross-ref**: CAND-COPY-MODE-SPECIFIC-TILE-BUDGET (mode-specific parallelism); OL-63. backend=ascendc.
