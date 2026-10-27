---
type: Runbook
title: "512B-aligned GM address yields 30% higher transfer bandwidth"
description: "For GM→UB transfers, 512B-aligned GM addresses have 30% higher bandwidth than 32B-aligned. When allocating output tensors in the pybind layer, ensure the start address is 512B-aligned (allocate with e"
tags: [task.optimization, topic.data_movement.data_copy, topic.memory, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-47-512b-aligned-gm-address-yields-30-higher-transfer.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a2]
aliases: ["OL-47"]
---

# 512B-aligned GM address yields 30% higher transfer bandwidth

> 原 OKF v1 卡号：**OL-47**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - output tensor allocation or DataCopy GM-offset computation

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Atlas A2 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
For GM→UB transfers, 512B-aligned GM addresses have 30% higher bandwidth than 32B-aligned. When allocating output tensors in the pybind layer, ensure the start address is 512B-aligned (allocate with extra padding then narrow). Note: this optimization is most effective on A2-series products.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-47（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
