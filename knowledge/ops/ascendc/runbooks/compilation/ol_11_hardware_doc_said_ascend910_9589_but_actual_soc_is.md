---
type: Runbook
title: "Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589"
description: "Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-11-hardware-doc-said-ascend910-9589-but-actual-soc-is.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-11"]
---

# Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589

> 原 OKF v1 卡号：**OL-11**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when configuring SOC_VERSION for CMake builds or kernel compilation

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when configuring SOC_VERSION for CMake builds or kernel compilation
- **Lesson**: The hardware documentation and initial setup guides listed the SOC as `Ascend910_9589`. The actual chip is `Ascend950PR_9589` -- the only Ascend variant that supports both SIMT and SIMD modes. Using the wrong SOC version causes silent compilation of kernels for the wrong architecture, which may run but produce incorrect results or suboptimal code. Always verify SOC version with `npu-smi info` on the actual hardware before trusting any documentation.
- **Evidence**: MEMORY.md ("Correct SOC version: Ascend950PR_9589"), docs/design/PLUGIN_PARADIGM_NOTES.md#ascend-chip-comparison

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-11（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
