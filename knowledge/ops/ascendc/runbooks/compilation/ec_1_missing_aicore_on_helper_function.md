---
type: Runbook
title: "Missing `__aicore__` on helper function"
description: "Missing `__aicore__` on helper function"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-1-missing-aicore-on-helper-function.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-1", "__aicore__"]
---

# Missing `__aicore__` on helper function

> 原 OKF v1 卡号：**EC-1**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: calling a __host__ function("helper_func") from a __aicore__ function("kernel_vf") is not allowed
  ```
- **Root cause**: All functions called inside `__simt_vf__ __aicore__` kernel VF functions must themselves be decorated with `__aicore__`. Bisheng treats undecorated functions as `__host__`-only, and cross-domain calls are forbidden.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  inline float compute_weight(float x) { return x * 0.5f; }

  // AFTER (compiles):
  __aicore__ inline float compute_weight(float x) { return x * 0.5f; }
  ```
- **Note**: Template helper functions also need `__aicore__`:
  ```cpp
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }
  ```
- **Related**: None (basic AscendC requirement)

<!-- 迁移自 porter kb/target/ascendc/（EC-1，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
