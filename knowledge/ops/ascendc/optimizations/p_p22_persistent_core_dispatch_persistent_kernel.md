---
type: Optimization
title: "Persistent-core dispatch (Persistent Kernel)"
description: "Anti-pattern: numBlocks = token_num (with many tokens, most blocks queue up time-slicing the 56 cores) cpp numBlocks = token_num; // e.g. 4096 blocks -> 56 cores queue 73 rounds for (int tid = threadI"
tags: [task.optimization, topic.implementation.atomic, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p22-persistent-core-dispatch-persistent-kernel.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Persistent-core dispatch (Persistent Kernel)

> 原 OKF v1 卡号：**P-P22**。

## 优化点

**Anti-pattern**: `numBlocks = token_num` (with many tokens, most blocks queue up time-slicing the 56 cores)
```cpp
numBlocks = token_num;  // e.g. 4096 blocks -> 56 cores queue 73 rounds
for (int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
    // handle 1 token
}
```

**Correct pattern**: 56 persistent blocks, each iterating over multiple tokens
```cpp
numBlocks = 56;  // MAX_AIV_CORES
for (uint32_t token = block_index; token < token_num; token += total_block_num) {
    // handle 1 token (same as original logic)
}
```

**Effect**: Eliminates block-scheduling overhead. medium (512 tokens) 3.2x speedup; xlarge (4096 tokens) 1.86x speedup.

**Applicability**:
- token_num >> MAX_AIV_CORES (56), with modest per-token work (scheduling overhead dominates)
- Forward-type kernels (no atomicAdd write conflicts)
- msprof scalar_ratio > 0.2 (significant indirect-addressing/scheduling-related code -> reducing scheduling is effective)
- **Does not apply** to backward (msprof vec_ratio ~= 1.0, compute-bound; scheduling overhead is not the bottleneck)

**Trigger condition (generator MUST check)**:
- `numBlocks = token_num` seen with token_num possibly >> 56 -> suggest generating a persistent variant
- msprof scalar_ratio > 0.2 -> persistent may be effective

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P22，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `thread_utilization`；trigger: work_items >> 56；优先级: MEDIUM。

> 索引行原文：`| P-P22 | Persistent kernel | thread_utilization | work_items >> 56 | MEDIUM |`
