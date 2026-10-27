---
type: Runbook
title: "README-driven doc updates"
description: "README-driven doc updates"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-10-readme-driven-doc-updates.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-10"]
---

# README-driven doc updates

> 原 OKF v1 卡号：**OL-10**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when creating or modifying any documentation file

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when creating or modifying any documentation file
- **Lesson**: Every documentation change must use README.md as the index and verify that all related documents are updated in sync. New files must be added to the README directory tree and document hierarchy. Batch results go into OPTIMIZATION_PLAN.md; README "current status" keeps only the final snapshot. This rule exists because Batch 9 changes were recorded in REPORT.md and EXPERT_FEEDBACK.md but README.md and OPTIMIZATION_PLAN.md were forgotten, leaving the project index stale.
- **Evidence**: CLAUDE.md lines 39-44, README.md directory tree

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-10（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
