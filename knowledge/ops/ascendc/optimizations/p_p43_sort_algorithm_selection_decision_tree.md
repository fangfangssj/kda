---
type: Optimization
title: "Sort Algorithm Selection Decision Tree"
description: "Problem: Different N/dtype/batch combinations need different sort algorithms. No single algorithm covers all scenarios. Decision tree (check in priority order): 1. N ≤ 4096 (fp32) or N ≤ 1024 (fp16/bf"
tags: [task.optimization, topic.memory, topic.tiling, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p43-sort-algorithm-selection-decision-tree.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Sort Algorithm Selection Decision Tree

> 原 OKF v1 卡号：**P-P43**。

## 优化点

**Problem**: Different N/dtype/batch combinations need different sort algorithms. No single algorithm covers all scenarios.

**Decision tree** (check in priority order):

```
1. N ≤ 4096 (fp32) or N ≤ 1024 (fp16/bf16)?
   YES → Hardware Merge Sort (Concat+Sort+Extract, P-P42)
   NO → continue

2. All data (value + index + tmp) fits in UB?
   YES → Single-core Radix Sort (AscendC Sort API with RADIX_SORT)
   NO → continue

3. B==1 AND 4096 < N ≤ 32768 AND fp32?
   YES → Multi-core Merge Sort (per-core local sort + 4-way hardware MrgSort merge)
   NO → continue

4. B ≥ core count AND N > 4096 AND blocksPerRow ≤ 256?
   YES → Big Batch Merge Sort (each core processes a whole row independently)
   NO → continue

5. Default → Multi-core Radix Sort (decoupled lookback prefix scan)
```

**Generator instructions**: When generating a Sort kernel:
1. Analyze the benchmark shape → determine N and dtype
2. For N ≤ 4096 (fp32): use P-P42 hardware sort directly — simplest and fastest
3. For N > 4096: consider radix sort, but implementation is complex. You can first use P-P42 on UB-sized chunks and then manually merge.

**Evidence**: CANN sort_tiling_arch35.cpp:746-761. E1 level.

**Stop condition**: When N < 32 the setup overhead dominates for all algorithms; a simple insertion sort may be faster.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P43，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `sort`；trigger: select sort algorithm by N/dtype/batch；优先级: HIGH。

> 索引行原文：`| P-P43 | Sort algorithm selection decision tree | sort | select sort algorithm by N/dtype/batch | HIGH |`
