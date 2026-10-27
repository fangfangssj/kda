---
type: Runbook
title: "Never mark issues \"confirmed safe\" without code change"
description: "Never mark issues \"confirmed safe\" without code change"
tags: [task.debugging, topic.precision, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-1-never-mark-issues-confirmed-safe-without-code-chan.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-1"]
---

# Never mark issues "confirmed safe" without code change

> 原 OKF v1 卡号：**OL-1**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - when reviewing generated code for correctness

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade. See `docs/data/kb_audit_cpu_truth_2026_04_29.md`.
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when reviewing generated code for correctness
- **Lesson**: int64 truncation was flagged 5 times by experts. The first 3 times, CC marked "confirmed safe" without changing code. The truncation was real -- `static_cast<int>(int64_var)` overflows when `dim * index > INT32_MAX`. It took 5 rounds and 72 code changes across 5 files to finally fix all instances.
- **Evidence**: EXPERT_FEEDBACK.md E8-3 (timeline of 5 rounds), CLAUDE.md lines 27-37

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-1（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
