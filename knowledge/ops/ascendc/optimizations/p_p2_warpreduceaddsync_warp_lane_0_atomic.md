---
type: Optimization
title: "WarpReduceAddSync + warp-lane-0 atomic"
description: "cpp // Anti-pattern: per-thread atomicAdd → 512 calls/block // Correct: warp reduce then lane-0 atomicAdd → 16 calls/block float warp_sum = Simt::WarpReduceAddSync(partial_sum); Simt::ThreadBarrier();"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p2-warpreduceaddsync-warp-lane-0-atomic.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# WarpReduceAddSync + warp-lane-0 atomic

> 原 OKF v1 卡号：**P-P2**。

## 优化点

```cpp
// Anti-pattern: per-thread atomicAdd → 512 calls/block
// Correct: warp reduce then lane-0 atomicAdd → 16 calls/block
float warp_sum = Simt::WarpReduceAddSync(partial_sum);
Simt::ThreadBarrier();
if (threadIdx.x % 32 == 0 && warp_sum != 0.0f)
    atomicAdd(dst, warp_sum);
```

**Precondition**: dst must be pre-zeroed on the host side. Measured SG backward: 0.865ms → 0.121ms (7.1x).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P2，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `scatter_add`；trigger: atomicAdd in reduction；优先级: HIGH。

> 索引行原文：`| P-P2 | WarpReduceAddSync | scatter_add | atomicAdd in reduction | HIGH |`
