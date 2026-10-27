---
type: Runbook
title: "VECIN-only pipeline cannot do GM→UB→GM pass-through"
description: "VECIN-only pipeline cannot do GM→UB→GM pass-through"
tags: [task.debugging, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.data_movement.data_copy, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-21-vecin-only-pipeline-cannot-do-gm-ub-gm-pass-throug.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-21"]
---

# VECIN-only pipeline cannot do GM→UB→GM pass-through

> 原 OKF v1 卡号：**EC-21**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only
- **Root cause**: VECIN syncs MTE2→VEC, but MTE3 store needs VEC→MTE3 sync. Without a VEC op and VECOUT queue, the pipeline has a sync gap.
- **Fix**: Split-queue pattern: VECIN for load + VECOUT for store + VEC identity op (Adds 0.0f) between them
- **Evidence**: Pad V2, Cat, Split all use this pattern (P-CAT-1)

<!-- 迁移自 porter kb/target/ascendc/（EC-21，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
