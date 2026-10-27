---
type: Runbook
title: "Performance benchmarking must use the target framework's standard tools"
description: "Performance benchmarking must use the target framework's standard tools"
tags: [task.optimization, topic.data_movement, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-31-performance-benchmarking-must-use-the-target-frame.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-31"]
---

# Performance benchmarking must use the target framework's standard tools

> 原 OKF v1 卡号：**OL-31**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - any performance report or benchmark-result publication

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process, measurement
- **Loaded by**: QA, Builder, Lead
- **Trigger**: any performance report or benchmark-result publication
- **Lesson**: In GELU evaluation, comparing C++ aclrtlaunch raw invocation against Python torch_npu invocation produced the wrong conclusion "CANN is 2-4x faster". After switching to NPUKernelBench's standard tool (utils/performance.py), results became 0.83-1.11x (all ≥0.8x PASS).
  **Root cause**: raw invocation has no Python dispatch overhead (~10us); torch_npu invocation does. The two timing baselines are not comparable.
  **Hard rules**:
  1. Use the target benchmark framework's standard evaluation tool; do not write your own timing code
  2. Ensure reference and candidate go through the same call path (same Python overhead)
  3. If you do write your own timing tool, document its differences from the standard tool
  4. Integration shape: kernel(.cpp/.h) + pybind11.cpp + model_new_ascendc.py(ModelNew)
  5. Build: `utils/build_ascendc.py`, Verify: `utils/verification_ascendc.py`, Perf: `utils/performance.py`
- **Evidence**: GELU evaluation (2026-04-08), NPUKernelBench framework analysis

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-31（category=process, measurement，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
