---
type: Runbook
title: "`LAUNCH_BOUND` value exceeds 512"
description: "`LAUNCH_BOUND` value exceeds 512"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-3-launch-bound-value-exceeds-512.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["EC-3", "LAUNCH_BOUND"]
---

# `LAUNCH_BOUND` value exceeds 512

> 原 OKF v1 卡号：**EC-3**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: 'LAUNCH_BOUND' attribute parameter 1024 exceeds maximum allowed value 512
  ```
  or at runtime: incorrect results / register spilling when LAUNCH_BOUND > 512 with complex kernel logic.
- **Root cause**: Ascend950PR supports LAUNCH_BOUND up to 2048 in theory, but **512 is the practical maximum** for kernels with non-trivial register usage. At 512 threads, each thread gets 64 registers (128KB register file / 512 threads / 4 bytes). Higher thread counts reduce per-thread registers, causing spills to slower memory and often incorrect codegen.
- **Fix**:
  ```cpp
  // BEFORE (risky or fails):
  LAUNCH_BOUND(1024) inline void kernel_vf(...) { ... }

  // AFTER (safe default):
  LAUNCH_BOUND(512) inline void kernel_vf(...) { ... }

  // Define as named constant:
  constexpr uint32_t OP_THREAD_NUM = 512;
  LAUNCH_BOUND(OP_THREAD_NUM) inline void kernel_vf(...) { ... }
  ```
- **Note**: source `__launch_bounds__(1024)` must be reduced to 512 when migrating. The dispatcher `Simt::Dim3{OP_THREAD_NUM}` must match the LAUNCH_BOUND value.
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK)

<!-- 迁移自 porter kb/target/ascendc/（EC-3，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
