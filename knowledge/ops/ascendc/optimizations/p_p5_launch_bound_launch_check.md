---
type: Optimization
title: "LAUNCH_BOUND + LAUNCH_CHECK"
description: "LAUNCH_BOUND(1024) >= the maximum number of threads any dispatcher may launch. Use LAUNCH_CHECK to inspect the return value of each launch."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p5-launch-bound-launch-check.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# LAUNCH_BOUND + LAUNCH_CHECK

> 原 OKF v1 卡号：**P-P5**。

## 优化点

`LAUNCH_BOUND(1024)` >= the maximum number of threads any dispatcher may launch. Use `LAUNCH_CHECK` to inspect the return value of each launch.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P5，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `kernel_launch`；trigger: every kernel；优先级: MEDIUM。

> 索引行原文：`| P-P5 | LAUNCH_BOUND + CHECK | kernel_launch | every kernel | MEDIUM |`
