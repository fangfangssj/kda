---
type: Optimization
title: "2D Tiling for Large Embedding Scatter"
description: "Problem: UB 256KB must simultaneously hold: index array + sort buffer + accumulation buffer + update data. When embDim is very large (e.g., embDim=16384, fp32 = 64KB/row), a single row occupies a larg"
tags: [task.optimization, topic.implementation.atomic, topic.memory, topic.tiling]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p41-2d-tiling-for-large-embedding-scatter.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# 2D Tiling for Large Embedding Scatter

> 原 OKF v1 卡号：**P-P41**。

## 优化点

**Problem**: UB 256KB must simultaneously hold: index array + sort buffer + accumulation buffer + update data. When embDim is very large (e.g., embDim=16384, fp32 = 64KB/row), a single row occupies a large UB footprint.

**Pattern**: Split [indicesNum, embDim] work into a 2D grid of rowTileNum × colTileNum:
1. Row partitioning: different cores handle different index ranges
2. Column partitioning: different embDim positions of the same index are handled by different cores
3. UB budget: `ubFactorRow * ubFactorCol * sizeof(T)` + index + sort buffer must be < UB size
4. Column-split cores finally merge results via atomicAdd (different columns of the same index are written to different GM addresses by different cores — no contention)

**Column-split threshold**: Only consider column splitting when `embDim * dtype_bytes > 4096`. Below that, fit the whole row into UB.

**Evidence**: CANN scatter_add_tiling_base.cpp:290-393 (FindUniqueCut + SimdTiling + DoBlockTiling). E1 level.

**Stop condition**: When embDim * dtype_bytes ≤ 4096, column splitting is not needed. Column splitting adds tiling complexity and should not be used for small-dim cases.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P41，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: embDim too large to fit with sort buffer in UB；优先级: MEDIUM。

> 索引行原文：`| P-P41 | 2D tiling for large embedding scatter | scatter_add | embDim too large to fit with sort buffer in UB | MEDIUM |`
