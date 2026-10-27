---
type: Optimization
title: "Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)"
description: "Problem: generate_assign_edges-class kernels use atomicCAS to detect first occurrence — one atomic per edge. When edge count is large, atomicCAS serialization becomes the bottleneck. Precondition: edg"
tags: [task.optimization, topic.implementation.pipeline, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p32-sorted-edge-first-occurrence-dedup-atomiccas-fre.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)

> 原 OKF v1 卡号：**P-P32**。

## 优化点

**Problem**: `generate_assign_edges`-class kernels use `atomicCAS` to detect first occurrence — one atomic per edge. When edge count is large, atomicCAS serialization becomes the bottleneck.

**Precondition**: edges are sorted by target (coordinate with P-P21 sort preprocessing).

**Anti-pattern** (atomicCAS per edge):
```cpp
// One atomicCAS per edge — checks whether this is the first write to the target
int old = Simt::AtomicCas(&assign_edges[target], INVALID, source);
if (old == INVALID) {
    // first occurrence
}
```

**Correct pattern** (adjacent comparison):
```cpp
// Precondition: edges sorted by edge_out
// Compare adjacent edges' targets — a change of target marks "first occurrence"
int prev_target = (tid > 0) ? edge_out[tid - 1] : -1;
int cur_target = edge_out[tid];
if (cur_target != prev_target) {
    // first occurrence of this target — no atomic needed!
    assign_edges[cur_target] = edge_in[tid];
}
```

**Advantages**:
- Zero atomics (pure load/store, no CAS contention)
- O(1) per element (vs atomicCAS's O(contention))
- Handles block boundaries: the first thread needs to compare with the previous block's last element (cross-block boundary)

**Measurement**: Pooling assign_edges: 259ms → 10ms (**25.6x**), combined with the overall sorted pipeline.

**Trigger condition**: any first-occurrence / dedup operation where the input is already sorted or can be pre-sorted.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P32，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: atomicCAS first-occurrence on sortable data；优先级: HIGH。

> 索引行原文：`| P-P32 | Sorted-edge dedup (atomicCAS-free) | scatter_add | atomicCAS first-occurrence on sortable data | HIGH |`
