---
type: Runbook
title: "int64 fixes cause ~6% performance regression on AscendC SIMT"
description: "Changing inner loop counters from `int` to `int64_t` (e.g., `for(int j)` → `for(int64_t j)`) increases register pressure on Ascend950PR. The AIV general-purpose registers are 32-bit; int64 occupies 2"
tags: [task.optimization, topic.programming_model.execution_model.simt, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-16-int64-fixes-cause-6-performance-regression-on-asce.md
    role: primary
created_at: "2026-07-10T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-16"]
---

# int64 fixes cause ~6% performance regression on AscendC SIMT

> 原 OKF v1 卡号：**OL-16**。
> 现象分类：perf_regression；观测置信度：inferred（推断）。
> 触发信号：
> - when applying int64 preservation fixes to AscendC SIMT kernels

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为推断沉淀（inferred），按卡内声明范围消费。

## 教训 / 根因
Changing inner loop counters from `int` to `int64_t` (e.g., `for(int j)` → `for(int64_t j)`) increases register pressure on Ascend950PR. The AIV general-purpose registers are 32-bit; int64 occupies 2 registers. Measured: Pooling D variant regressed 24.55ms → 25.94ms (~6%). The int64 interface guarantee (CLAUDE.md) is correct and must not be violated, but the regression is expected and must be reported honestly. Do NOT claim "no perf regression" without benchmark verification.

## 证据
E9-1 investigation (2026-03-29), commit 475e83c claimed "no perf regression" without running benchmark

<!-- LLM-辅助分类迁移(convert 二次 pass, migrate_ol_dispositions.py)。confidence=inferred,待人工确认。原 OL-16。 -->
