---
type: Runbook
title: "bisheng multi-branch Simt::VF_CALL causes 507035 crash"
description: "Putting multiple `Simt::VF_CALL<template1>(...); Simt::VF_CALL<template2>(...)` branches inside one `extern \"C\" __global__` function can cause ACL ERR 507035 (kernel launch failure). Root cause: bishe"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-17-bisheng-multi-branch-simt-vf-call-causes-507035-cr.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-17"]
---

# bisheng multi-branch Simt::VF_CALL causes 507035 crash

> 原 OKF v1 卡号：**OL-17**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when writing AscendC __global__ dispatcher with multiple Simt::VF_CALL template instantiations

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Putting multiple `Simt::VF_CALL<template1>(...); Simt::VF_CALL<template2>(...)` branches inside one `extern "C" __global__` function can cause ACL ERR 507035 (kernel launch failure). Root cause: bisheng compiler/linker binary slot corruption — certain kernel slot positions in a large binary produce broken device code. Fix: split each template instantiation into its own `__global__` entry point. Host-side dispatch selects the right entry.

## 证据
Pooling B variant crashed on cluster 2 (dim=1, edges=1024). After splitting into separate entry points, all 61 clusters pass.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-17（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
