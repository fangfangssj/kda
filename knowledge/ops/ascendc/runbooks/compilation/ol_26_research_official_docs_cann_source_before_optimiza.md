---
type: Runbook
title: "Research official docs + CANN source before optimization attempts"
description: "Research official docs + CANN source before optimization attempts"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-26-research-official-docs-cann-source-before-optimiza.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-26"]
---

# Research official docs + CANN source before optimization attempts

> 原 OKF v1 卡号：**OL-26**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - before any non-trivial optimization or architecture change

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, Optimizer, Researcher
- **Trigger**: before any non-trivial optimization or architecture change
- **Lesson**: E14 first attempt was a guess (TQue input + TBuf accum). If we had first verified TBuf sync semantics from official docs, we would have known it can't work. The correct pattern (accum in TQue<VECOUT>) was already in the forward code and confirmed by CANN MoE source.
  Research workflow:
  1. Check official docs (dev-browser for JS-rendered hiascend.com)
  2. Check CANN source code (~/workspace/cann/, git fetch first)
  3. Record findings as patterns/OL entries for future use
  4. THEN implement
- **Evidence**: E14 session (2026-04-06 → 2026-04-07), ASCENDC_LANGUAGE_REFERENCE.md creation

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-26（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
