---
type: Runbook
title: "bf16 Adds with a zero scalar may hit PB-4"
description: "`static_cast<bfloat16_t>(0)` triggers PB-4 (bisheng bf16 scalar cast bug), so Adds does not actually add 0. **Safe alternatives**: (1) fp32 round-trip: `Cast(fp32, bf16, CAST_NONE) → modify → Cast(bf1"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-74-bf16-adds-with-a-zero-scalar-may-hit-pb-4.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-74"]
---

# bf16 Adds with a zero scalar may hit PB-4

> 原 OKF v1 卡号：**OL-74**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - wanting to use `Adds(dst, src, static_cast<bfloat16_t>(0), count)` for bf16 data movement (bridge / no-op pass-through)

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
`static_cast<bfloat16_t>(0)` triggers PB-4 (bisheng bf16 scalar cast bug), so Adds does not actually add 0. **Safe alternatives**: (1) fp32 round-trip: `Cast(fp32, bf16, CAST_NONE) → modify → Cast(bf16, fp32, CAST_ROUND)`; (2) for identity copy, use OL-73 pybind clone(); (3) use `Mul(dst, src, one_buf, count)` where one_buf is pre-filled with 1.0; (4) use `Duplicate(dst, src_value, count)` combined with GetValue for scalar extraction (still beware bf16 Cast). **Best practice**: never use bf16 scalar literals in arithmetic; convert to fp32 first for all bf16 math.

## 证据
29_TanhGatedResidualAddBackward first tried `Adds(dst, src, bf16(0))` for copy — compiled but numerically wrong at runtime. Switching to fp32 round-trip + clone() → 50/50 PASS. Related to PB-4. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-74（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
