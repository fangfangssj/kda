---
type: Runbook
title: "Use aclrtEvent not chrono for NPU timing; memset outside event"
description: "Two measurement errors compounded in early benchmarks. (1) `std::chrono` measures wall-clock time including host synchronization overhead, not device execution time -- use `aclrtCreateEvent` / `aclrtR"
tags: [task.optimization, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-5-use-aclrtevent-not-chrono-for-npu-timing-memset-ou.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-5"]
---

# Use aclrtEvent not chrono for NPU timing; memset outside event

> 原 OKF v1 卡号：**OL-5**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - when writing or reviewing NPU benchmark code

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 教训 / 根因
Two measurement errors compounded in early benchmarks. (1) `std::chrono` can
capture host launch/synchronization effects instead of device execution; use
`aclrtCreateEvent` / `aclrtRecordEvent` / `aclrtEventElapsedTime`. (2)
`aclrtMemset` was inside the timed region and inflated backward latency. Put
initialization outside the event window and discard conclusions produced by the
legacy timing path.

## 证据
BENCHMARK_METHODOLOGY.md Section 1, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md ("旧计时方法" / legacy timing method)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-5（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
