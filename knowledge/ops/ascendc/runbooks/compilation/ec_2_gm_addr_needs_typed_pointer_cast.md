---
type: Runbook
title: "`GM_ADDR` needs typed pointer cast"
description: "`GM_ADDR` needs typed pointer cast"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-2-gm-addr-needs-typed-pointer-cast.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-2", "GM_ADDR"]
---

# `GM_ADDR` needs typed pointer cast

> 原 OKF v1 卡号：**EC-2**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: cannot initialize a variable of type '__gm__ float *' with an lvalue of type 'GM_ADDR' (aka 'uint8_t * __attribute__((address_space(1)))')
  ```
  or:
  ```
  error: subscript of pointer to type '__gm__ uint8_t' ... is not allowed
  ```
- **Root cause**: `GM_ADDR` is `__gm__ uint8_t*`. Kernel VF functions receive all GM pointers as untyped `GM_ADDR`. To access data as a specific type, you must cast with `reinterpret_cast<__gm__ T*>`. The `__gm__` qualifier must be preserved through the cast.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  __gm__ float* input = input_gm;              // type mismatch
  float val = input_gm[i];                      // subscript on uint8_t*

  // AFTER (compiles):
  __gm__ float* input = reinterpret_cast<__gm__ float*>(input_gm);
  float val = input[i];                          // correct typed access

  // For const pointers:
  __gm__ const int* edge_in = reinterpret_cast<__gm__ const int*>(edge_in_gm);
  ```
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK — kernel launch pattern)

<!-- 迁移自 porter kb/target/ascendc/（EC-2，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
