---
type: Runbook
title: "`simt_compat.h` conflicts in NPU mode"
description: "`simt_compat.h` conflicts in NPU mode"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-4-simt-compat-h-conflicts-in-npu-mode.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-4", "simt_compat.h"]
---

# `simt_compat.h` conflicts in NPU mode

> 原 OKF v1 卡号：**EC-4**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: redefinition of 'blockDim' as different kind of symbol
  ```
  or:
  ```
  error: expected unqualified-id
  ```
  (when `#define blockDim` macro clashes with CANN's built-in `blockDim` in NPU mode)
- **Root cause**: `simt_compat.h` defines `blockDim` and `threadIdx` as macros that map to raw CPU-mode globals (`g_threadDimX`, `g_threadIdxX`). In NPU mode, CANN provides its own built-in `blockDim`/`threadIdx` — the macros collide with these built-ins. The header must only be included in CPU debug builds.
- **Fix**:
  ```cpp
  // BEFORE (fails on NPU):
  #include "simt_compat.h"    // unconditional include → macro conflicts

  // AFTER (conditional):
  #if defined(ASCENDC_CPU_DEBUG)
  #include "simt_compat.h"
  #endif
  ```
  The guard works because:
  - CPU debug mode: `ASCENDC_CPU_DEBUG` is defined by tikicpulib CMake target → macros active
  - NPU mode: `ASCENDC_CPU_DEBUG` is not defined → header skipped, CANN built-ins used
- **Related**: None (project-specific compatibility layer)

<!-- 迁移自 porter kb/target/ascendc/（EC-4，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
