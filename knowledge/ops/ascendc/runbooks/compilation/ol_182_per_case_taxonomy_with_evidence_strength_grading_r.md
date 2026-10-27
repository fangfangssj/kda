---
type: Runbook
title: "Per-case taxonomy with evidence_strength grading required before claiming fix coverage"
description: "applies_to: soc=all; cann=all; op_class=all; workflow=verification"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-182-per-case-taxonomy-with-evidence-strength-grading-r.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-182"]
---

# Per-case taxonomy with evidence_strength grading required before claiming fix coverage

> 原 OKF v1 卡号：**OL-182**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=all; cann=all; op_class=all; workflow=verification

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; op_class=all; workflow=verification`
`verified_on: multiple ops (26_AvgPool3d, 27_MaxPool3d)`

**Principle**: Derived from FA V220 decision discussion (2026-05-22). Claims like "X of Y cases solvable" without per-case evidence and strength grading (HIGH/MEDIUM/LOW) are the same shape as "0.014× CANN is real ceiling" — confident framing without per-instance evidence. LOW-confidence entries must be classified `OTHER`, not the predicted-fix bucket.

**Application**: Before declaring ko-2 fixes for residual failures, produce `fixture_eval_err_taxonomy.json` with per-case root cause classification + evidence_strength. This prevents scope substitution.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-182（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
