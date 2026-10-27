---
type: Runbook
title: "Every code change MUST be followed by BOTH precision AND performance verification"
description: "Every code change MUST be followed by BOTH precision AND performance verification"
tags: [task.debugging, topic.precision, topic.data_movement, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-18-every-code-change-must-be-followed-by-both-precisi.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-18"]
---

# Every code change MUST be followed by BOTH precision AND performance verification

> 原 OKF v1 卡号：**OL-18**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - after any kernel code modification

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: after any kernel code modification
- **Lesson**: CLAUDE.md said "every code update must verify precision" but did not mention performance. int64 fixes passed precision (61/61 PASS) but caused 6% performance regression that went undetected for 2 batches. The rule must be: verify precision AND performance after every change. A commit message claiming "no regression" without benchmark data is a lie.
- **Evidence**: E9-1 investigation, REPORT.md data was stale for 2 batches

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-18（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
