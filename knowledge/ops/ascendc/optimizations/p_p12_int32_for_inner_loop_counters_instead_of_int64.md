---
type: Optimization
title: "int32 for inner-loop counters instead of int64"
description: "Use int for variables whose range is within int32 (loop counter j, thread_idx_emb, etc.). edge_in[i] emb_dim must stay int64. Effect: fwd -3%, bwd -2%."
tags: [task.optimization, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p12-int32-for-inner-loop-counters-instead-of-int64.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# int32 for inner-loop counters instead of int64

> 原 OKF v1 卡号：**P-P12**。

## 优化点

Use `int` for variables whose range is within int32 (loop counter j, thread_idx_emb, etc.). `edge_in[i] * emb_dim` must stay int64. Effect: fwd -3%, bwd -2%.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P12，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `memory_access`；trigger: int64 in hot loop；优先级: MEDIUM。

> 索引行原文：`| P-P12 | int32 inner loop | memory_access | int64 in hot loop | MEDIUM |`
