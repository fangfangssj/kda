---
type: Runbook
title: "Expert feedback must result in code changes, never just documentation"
description: "Expert feedback must result in code changes, never just documentation"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-2-expert-feedback-must-result-in-code-changes-never.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-2"]
---

# Expert feedback must result in code changes, never just documentation

> 原 OKF v1 卡号：**OL-2**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - when processing expert review feedback

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when processing expert review feedback
- **Lesson**: Marking expert-reported issues as "already safe" in documentation -- without modifying, compiling, and testing the code -- is forbidden. The int64 problem was annotated "confirmed safe" in docs 3 times while the truncating `static_cast<int>()` calls remained in the source. The rule is: modify code + compile + run precision tests, or the issue stays open.
- **Evidence**: CLAUDE.md lines 34-37, EXPERT_FEEDBACK.md E8-3 round-by-round history

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-2（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
