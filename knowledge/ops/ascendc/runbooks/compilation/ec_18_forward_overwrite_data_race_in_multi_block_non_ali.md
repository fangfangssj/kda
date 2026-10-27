---
type: Runbook
title: "Forward-overwrite data race in multi-block non-aligned DMA"
description: "Forward-overwrite data race in multi-block non-aligned DMA"
tags: [task.debugging, topic.precision, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-18-forward-overwrite-data-race-in-multi-block-non-ali.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-18"]
---

# Forward-overwrite data race in multi-block non-aligned DMA

> 原 OKF v1 卡号：**EC-18**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail).

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail). When multiple blocks process different rows in parallel, the overwrites from different blocks race.
- **Root cause**: Block K-1's tail overwrite extends into Block K's write region. Without ordering, Block K may read before K-1's overwrite completes, or K's write may be overwritten by K-1's stale data.
- **Fix**: Two approaches:
  1. **Per-row overlap** (chunk >= ALIGN): re-copy last ALIGN elements from `chunk - ALIGN` offset. No cross-row overwrite. Safe for multi-block.
  2. **nblk=1 + padded alloc** (chunk < ALIGN): serialize to one block. Over-allocate output with ALIGN padding, narrow() after kernel.
- **Evidence**: Split V3 — V2 forward-overwrite caused 12 new failures, fixed with per-row overlap (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-18，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
