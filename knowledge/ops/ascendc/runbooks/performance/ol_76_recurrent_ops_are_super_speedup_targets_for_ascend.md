---
type: Runbook
title: "Recurrent ops are super-speedup targets for AscendC"
description: "The reference for Python-looped ops (RWKV WKV, recurrent state update, sequential scan) has one torch-op launch overhead per step (~100us/step). At seq_len=100 that is ~10ms of pure launch overhead. A"
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.memory, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-76-recurrent-ops-are-super-speedup-targets-for-ascend.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-76"]
---

# Recurrent ops are super-speedup targets for AscendC

> 原 OKF v1 卡号：**OL-76**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - reference implementation has a Python `for t in range(seq_len)` loop, with each step calling an independent torch op

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
The reference for Python-looped ops (RWKV WKV, recurrent state update, sequential scan) has one torch-op launch overhead per step (~100us/step). At seq_len=100 that is ~10ms of pure launch overhead. An AscendC single-kernel internal loop eliminates **all** per-step launch overhead, keeping state in UB without GM round-trip. Speedup ≈ O(seq_len × launch_overhead / kernel_compute_time). **This is not compute optimization — it is launch elimination.** Even with no VEC tuning inside AscendC, replacing the Python loop with an in-kernel C++ loop yields 10-50x.

## 证据
30_TimeDecayExponentialStabilization (RWKV): Python ref mean 4.56ms (mix of seq=1..100), AscendC mean 0.13ms → **29.7x mean** (independent verification). Worst 4.3x (seq=1, almost no loop), best 49.8x (seq=100, loop dominates). 50/50 ≥ 1.0x. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-76（category=performance，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
