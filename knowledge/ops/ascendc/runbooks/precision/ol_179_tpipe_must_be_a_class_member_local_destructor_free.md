---
type: Runbook
title: "TPipe must be a class member — local destructor frees TQue buffers before async MTE3 writes complete"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-179-tpipe-must-be-a-class-member-local-destructor-free.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-179"]
---

# TPipe must be a class member — local destructor frees TQue buffers before async MTE3 writes complete

> 原 OKF v1 卡号：**OL-179**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all`
`verified_on: soc=Ascend910_9382 (V220); cann=9.0.0`

**Principle**: TPipe manages TQue buffer allocation. When declared as local variable, its destructor runs at scope exit, potentially freeing TQue buffers before MTE3 finishes writing to GM. When TPipe is a class member, it persists for the kernel object's lifetime.

**Evidence**: 12_Permute a3-ds kw-3 (2026-05-21). No precision failures attributable to TPipe lifecycle when member.

**Cross-ref**: OL-94 (TQue vs TBuf sync decision table); OL-128 (V220 TBuf→MTE3 stale data).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-179（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
