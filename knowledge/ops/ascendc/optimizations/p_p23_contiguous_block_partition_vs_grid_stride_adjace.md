---
type: Optimization
title: "Contiguous-block partition vs grid-stride (adjacent-element comparison scenario)"
description: "Problem: when comparing adjacent elements (arr[i] vs arr[i-1]), a grid-stride loop causes every read of arr[i-1] to be a cache miss (distance total_threads elements, typically 28672). Anti-pattern (gr"
tags: [task.optimization, topic.implementation.atomic, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p23-contiguous-block-partition-vs-grid-stride-adjace.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Contiguous-block partition vs grid-stride (adjacent-element comparison scenario)

> 原 OKF v1 卡号：**P-P23**。

## 优化点

**Problem**: when comparing adjacent elements (`arr[i] vs arr[i-1]`), a grid-stride loop causes every read of `arr[i-1]` to be a cache miss (distance `total_threads` elements, typically 28672).

**Anti-pattern** (grid-stride, cache miss):
```cpp
for (int64_t i = tid; i < n; i += total_threads) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] is 28672 elements from arr[i] in HBM
}
```

**Correct pattern** (contiguous block, cache-friendly):
```cpp
int64_t chunk = (n + total_threads - 1) / total_threads;
int64_t start = tid * chunk;
int64_t end = min(start + chunk, n);
for (int64_t i = start; i < end; i++) {
    if (i == 0 || arr[i] != arr[i - 1]) { ... }  // arr[i-1] adjacent to arr[i]
}
```

**Measured**: assign_edges sorted scan from 123ms → 10ms (**12x**). Total optimization (including atomicCAS avoidance): 259ms → 10ms (**25.6x**).

**Trigger condition**: adjacent access `arr[i-1]` or `arr[i+1]` inside the loop + grid-stride loop → switch to contiguous block.

**Note**: contiguous-block partition is not appropriate for all scenarios. If each iteration's data is fully independent (no adjacent dependency), grid-stride's coalesced access is instead better. Use contiguous block only when **adjacent-element comparison / dependency is required**.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P23，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: arr[i-1] neighbor access in loop；优先级: HIGH。

> 索引行原文：`| P-P23 | Contiguous-chunk vs grid-stride | memory_access | arr[i-1] neighbor access in loop | HIGH |`
