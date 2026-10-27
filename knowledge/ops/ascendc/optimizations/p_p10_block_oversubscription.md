---
type: Optimization
title: "Block Oversubscription"
description: "nblk > physical AIV core count (56) → disperses atomicAdd contention. Applicability conditions (key): Only effective when the kernel has atomicAdd contention. - Unsorted backward (has atomicAdd conten"
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.atomic, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p10-block-oversubscription.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Block Oversubscription

> 原 OKF v1 卡号：**P-P10**。

## 优化点

nblk > physical AIV core count (56) → disperses atomicAdd contention.

**Applicability conditions (key)**: Only effective when the kernel has atomicAdd contention.
- **Unsorted backward** (has atomicAdd contention): nblk=224 → bwd 2.0x speedup ✅
- **Sorted backward** (register accumulation, no atomicAdd): nblk=56 is strictly optimal. Oversubscription monotonically gets worse: 112 (+3.8%), 224 (+9.8%), 448 (+20.8%) ❌
- **Forward**: nblk=56 is always optimal (pipe-bound; oversubscription gives no benefit)

**Root cause**: Sorted + register accumulation eliminates atomicAdd contention → the benefit of dispersing contention via oversubscription disappears → only the register-pressure cost remains. Multiple blocks compete for the same AIV core's register file, and accumulator variables get spilled to HBM.

**E9-2 measurement confirmed** (2026-03-30, 61 clusters, NPU idle): see E10 exploration results.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P10，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: atomicAdd contention；优先级: HIGH。

> 索引行原文：`| P-P10 | Block oversubscription | scatter_add | atomicAdd contention | HIGH |`
