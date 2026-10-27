---
type: Optimization
title: "Fan-In Threshold for Sort Decision (CANN-derived)"
description: "Problem: P-P21 describes the sort + register-accumulation pattern but does not give a quantitative threshold for when sorting should be enabled. Sorting has overhead (O(N log N)); at low fan-in it is"
tags: [task.optimization, topic.implementation.atomic, topic.tiling, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p39-fan-in-threshold-for-sort-decision-cann-derived.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Fan-In Threshold for Sort Decision (CANN-derived)

> 原 OKF v1 卡号：**P-P39**。

## 优化点

**Problem**: P-P21 describes the sort + register-accumulation pattern but does not give a quantitative threshold for when sorting should be enabled. Sorting has overhead (O(N log N)); at low fan-in it is actually slower.

**Decision rule** (CANN scatter_add_tiling_base.cpp):
- `fan_in_ratio = indicesNum / varShape[0]` (input index count / output row count)
- **fan_in > 10:1** → enable sort (general threshold)
- **fan_in > 3:1 AND embDim * dtype_bytes >= 100K** → also enable sort (at large dim each atomicAdd is more expensive, lowering the threshold for sorting)
- **fan_in ≤ 3:1** → do not sort; atomicAdd directly

**Generator must check**: when generating a scatter-add kernel, analyze the benchmark's shape spec:
1. Compute fan_in_ratio = index_count / unique_target_count
2. fan_in > 10 → generate a variant with sort
3. fan_in 3-10 and embDim is large → generate a variant with sort
4. fan_in < 3 → baseline atomicAdd is sufficient

**Evidence**: CANN scatter_add_tiling_base.cpp:204-211 (`isSort_ = indicesNum_ > varShape_[0] * TEN`), E1 level (inferred from source).

**Stop condition**: when fan_in < 3 the sort overhead exceeds the savings. When index is already pre-sorted, skip sort and accumulate directly.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P39，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: scatter-add fan-in ratio determines whether to sort；优先级: HIGH。

> 索引行原文：`| P-P39 | Fan-in threshold for sort decision | scatter_add | scatter-add fan-in ratio determines whether to sort | HIGH |`
