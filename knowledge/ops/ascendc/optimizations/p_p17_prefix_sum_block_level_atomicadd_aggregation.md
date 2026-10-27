---
type: Optimization
title: "Prefix-sum + block-level atomicAdd aggregation"
description: "Three-level aggregation to reduce global atomics: 1. In-group prefix sum (__shfl_up): each thread obtains local_offset 2. Group leader → UB atomicAdd (__ubuf__): once per group (512/32 = 16 times) 3."
tags: [task.optimization, topic.implementation.atomic, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p17-prefix-sum-block-level-atomicadd-aggregation.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Prefix-sum + block-level atomicAdd aggregation

> 原 OKF v1 卡号：**P-P17**。

## 优化点

Three-level aggregation to reduce global atomics:
1. **In-group prefix sum** (`__shfl_up`): each thread obtains local_offset
2. **Group leader → UB atomicAdd** (`__ubuf__`): once per group (512/32 = 16 times)
3. **Block leader → global atomicAdd**: once per whole block

512 global atomicAdd → 1. **Directly applicable to Pooling backward atomicAdd optimization.**

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P17，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: scatter-add aggregation；优先级: HIGH。

> 索引行原文：`| P-P17 | Prefix-sum + block atomic | scatter_add | scatter-add aggregation | HIGH |`
