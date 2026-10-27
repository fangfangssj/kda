---
type: Runbook
title: "torch::zeros on NPU is NOT stream-ordered with custom kernels"
description: "The zero-fill of `torch::zeros` is not necessarily executed on the same stream as `aclrtlaunch_*`. A custom kernel may begin before zero-fill finishes, reading garbage data. **Fix**: use `torch::empty"
tags: [task.debugging, topic.implementation.synchronization, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-66-torch-zeros-on-npu-is-not-stream-ordered-with-cust.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-66"]
---

# torch::zeros on NPU is NOT stream-ordered with custom kernels

> 原 OKF v1 卡号：**OL-66**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - custom AscendC kernel uses a torch::zeros tensor as output

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
The zero-fill of `torch::zeros` is not necessarily executed on the same stream as `aclrtlaunch_*`. A custom kernel may begin before zero-fill finishes, reading garbage data. **Fix**: use `torch::empty` + `aclrtMemsetAsync(stream)` instead of `torch::zeros`. Or use two kernel launches (the second implicitly waits for the first).

## 证据
Histc debugging 2026-04-14, 5/5 failures without fix, 5/5 passes with fix. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-66（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
