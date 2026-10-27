---
type: Optimization
title: "Adaptive tile size"
description: "dim ≤ 256: <BRE=32, TI=16> 16 edges × 32 emb threads dim ≤ 512: <BRE=64, TI=8> 8 edges × 64 emb threads dim > 512: <BRE=512, TI=1> 1 edge × 512 emb threads NPU 56 blocks × TI=16 = 896 edges/step → TI="
tags: [task.optimization, topic.data_movement.data_copy, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p11-adaptive-tile-size.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Adaptive tile size

> 原 OKF v1 卡号：**P-P11**。

## 优化点

```
dim ≤ 256:  <BRE=32,  TI=16>    16 edges × 32 emb threads
dim ≤ 512:  <BRE=64,  TI=8>     8 edges × 64 emb threads
dim > 512:  <BRE=512, TI=1>     1 edge × 512 emb threads
```

On a 56-AIV target, 56 blocks × TI=16 covers 896 edges per step. Tune TI
from the target workload and UB budget instead of copying a fixed launch shape.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P11，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: multi-dim tiling；优先级: HIGH。

> 索引行原文：`| P-P11 | Adaptive tile size | memory_access | multi-dim tiling | HIGH |`
