---
type: Runbook
title: "Precision evidence: CPU truth and source-arch behavior capture"
description: "CPU fp64 anchors the declared math while an authorized source-arch run records observable migration behavior; target prior-art may inform generation but is not final truth."
tags: [task.debugging, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-97-precision-target-cpu-truth-vs-cann-bit-match.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-29T00:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-97"]
---

# Precision evidence: CPU truth and source-arch behavior capture

> 原 OKF v1 卡号：**OL-97**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - the generated result agrees with CPU truth but differs from the source-arch capture, or vice versa

## 现象 / 触发

A generated operator agrees with CPU fp64 truth but differs from an authorized arch22 behavior
capture, or agrees with the capture but violates the declared equation.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

Treat these as complementary evidence channels. CPU fp64 anchors the mathematical contract. A run
of the user-selected operation on the source-arch NPU records observable migration behavior,
including attributes, optional inputs, output structure, and numerical conventions. Record the
reference source in the manifest and classify disagreements by region, reduction order, and dtype.

An existing arch35 implementation, installed target operator, target binary, profile, or archive
may be consulted as prior-art for generation and research. It is not migration or backward truth;
thresholds must come from the declared operator contract rather than an unrelated verifier.
