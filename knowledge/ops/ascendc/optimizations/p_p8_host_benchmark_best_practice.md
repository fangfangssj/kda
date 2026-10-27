---
type: Optimization
title: "Host benchmark best practice"
description: "Warm up 3 times, time 10 NPU iterations, compare precision with audited CPU truth, then run boundary tests."
tags: [task.optimization, topic.precision, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p8-host-benchmark-best-practice.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Host benchmark best practice

> 原 OKF v1 卡号：**P-P8**。

## 优化点

Warm up 3 times, time 10 NPU iterations and take the mean, compare precision
against audited CPU truth, then run boundary tests (`edges=0`, `dim=1`, `dim=3`).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P8，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `kernel_launch`；trigger: benchmark code；优先级: LOW。

> 索引行原文：`| P-P8 | Host benchmark practice | kernel_launch | benchmark code | LOW |`
