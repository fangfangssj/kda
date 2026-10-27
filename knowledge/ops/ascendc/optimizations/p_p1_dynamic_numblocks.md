---
type: Optimization
title: "Dynamic numBlocks"
description: "cpp // Pooling: saturate all AIV cores constexpr uint32_t MAX_AIV_CORES = 56; // 28 AICore x 2 AIV // SG: each block handles one token uint32_t fwd_blk = token_num grid_y; Distinction: Pooling shares"
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p1-dynamic-numblocks.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Dynamic numBlocks

> 原 OKF v1 卡号：**P-P1**。

## 优化点

```cpp
// Pooling: saturate all AIV cores
constexpr uint32_t MAX_AIV_CORES = 56;  // 28 AICore x 2 AIV

// SG: each block handles one token
uint32_t fwd_blk = token_num * grid_y;
```

**Distinction**: Pooling shares work by stride -> saturate all cores. SG has each block own one output slice -> numBlocks = number of work items.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P1，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `thread_utilization`；trigger: any kernel launch；优先级: HIGH。

> 索引行原文：`| P-P1 | numBlocks dynamic | thread_utilization | any kernel launch | HIGH |`
