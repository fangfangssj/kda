---
type: Optimization
title: "vec4 vectorization path enablement condition"
description: "Anti-pattern: if (hidden_dim % 4 == 0 && grid_y > 1) — grid_y > 1 is a redundant gate. Correct: if (hidden_dim % 4 == 0) — when grid_y==1, block_y=0 is entirely correct."
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p3-vec4-vectorization-path-enablement-condition.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# vec4 vectorization path enablement condition

> 原 OKF v1 卡号：**P-P3**。

## 优化点

**Anti-pattern**: `if (hidden_dim % 4 == 0 && grid_y > 1)` — `grid_y > 1` is a redundant gate.

**Correct**: `if (hidden_dim % 4 == 0)` — when grid_y==1, block_y=0 is entirely correct.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P3，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: dim % 4 == 0；优先级: MEDIUM。

> 索引行原文：`| P-P3 | vec4 vectorization | memory_access | dim % 4 == 0 | MEDIUM |`
