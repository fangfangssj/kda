---
type: Runbook
title: "Audited CPU PyTorch is the numerical specification"
description: "Audited CPU PyTorch is the numerical specification"
tags: [task.debugging, topic.precision, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-28-audited-cpu-pytorch-is-the-numerical-specification.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["OL-28"]
---

# Audited CPU PyTorch is the numerical specification

> 原 OKF v1 卡号：**OL-28**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - any arch22→arch35 migration or forward→backward generation task

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ **VALIDATED-CPU — this entry is the anchor for the CPU-truth approach**. PyTorch CPU is the specification, not any single device implementation. Aligns with OL-97.
- **Category**: process, trust_calibration
- **Loaded by**: Analyzer, Builder, QA, Lead
- **Trigger**: any arch22→arch35 migration or forward→backward generation task
- **Lesson**: device implementations may contain rounding or edge-case behavior that differs from the mathematical contract. The target implementation must be checked against an audited CPU specification.
  **The verification chain should be**:
  ```
  PyTorch definition (spec, mathematical contract)
  AscendC implementation
      ↓ verify
  bit-exact or within-tolerance against the PyTorch spec
  ```
  **Hard rules**:
  1. Build or audit the CPU forward specification before implementation.
  2. Derive backward truth from the same forward specification.
  3. Record any arch22/CANN versus CPU differences as diagnostics, never as a replacement truth.
  4. Verify arch35 outputs and every requested gradient against CPU truth.
- **Evidence**: project precision-audit practice summarized by OL-97.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-28（category=process, trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
