---
type: Runbook
title: "Always verify expert claims with hardware test"
description: "Always verify expert claims with hardware test"
tags: [task.optimization, topic.data_movement, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-7-always-verify-expert-claims-with-hardware-test.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-7"]
---

# Always verify expert claims with hardware test

> 原 OKF v1 卡号：**OL-7**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - when an expert makes a hardware performance claim without providing measurement data

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: trust_calibration
- **Loaded by**: Optimizer, Planner
- **Trigger**: when an expert makes a hardware performance claim without providing measurement data
- **Lesson**: An expert stated that 128-bit loads are slower than 64-bit on Ascend950PR. Hardware testing (`tests/load_width_test/`) showed the opposite: 128-bit is 1.2x-2.1x faster than 32-bit for sequential reads (1MB: 1.17x, 16MB: 1.39x, 64MB: 2.12x). The expert's claim may have been valid for a different access pattern (random/indirect indexing), but for the sequential reads in our kernels, wider loads are strictly better. Always run a targeted micro-benchmark before accepting or rejecting a hardware claim.
- **Evidence**: EXPERT_FEEDBACK.md E7-3 (128-bit实测数据 / 128-bit measured), `tests/load_width_test/`

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-7（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
