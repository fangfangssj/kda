---
type: Runbook
title: "bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)"
description: "AscendC's `CAST_ROUND` and `CAST_RINT` are **different** rounding modes: - `CAST_ROUND`: **round half UP** — if the first bit of the fraction being rounded is 1, round up (i.e. 0.5 always rounds up) -"
tags: [task.debugging, topic.precision]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-81-bf16-fp16-cast-default-should-be-cast-rint-not-cas.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-81"]
---

# bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)

> 原 OKF v1 卡号：**OL-81**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel uses `Cast(dst, src, RoundMode::CAST_ROUND, count)` for fp32→bf16 or fp32→fp16 and needs to bit-exactly match the torch_npu/PyTorch reference

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
AscendC's `CAST_ROUND` and `CAST_RINT` are **different** rounding modes:
  - `CAST_ROUND`: **round half UP** — if the first bit of the fraction being rounded is 1, round up (i.e. 0.5 always rounds up)
  - `CAST_RINT`: **IEEE 754 RNE** — round half to even (0.5 decided by parity of the last kept bit)
  - torch_npu / PyTorch / almost all IEEE 754 hardware defaults to **RNE**
  - **Therefore, to align with PyTorch bf16/fp16 behavior you must use CAST_RINT, not CAST_ROUND**

## 证据
E2 level (empirical: a single flag flip raised bf16 from 1/9 PASS to 8/9 PASS)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-81（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
