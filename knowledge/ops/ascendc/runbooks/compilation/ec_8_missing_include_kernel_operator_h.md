---
type: Runbook
title: "Missing `#include <kernel_operator.h>`"
description: "Missing `#include <kernel_operator.h>`"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.implementation.atomic, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-8-missing-include-kernel-operator-h.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-8"]
---

# Missing `#include <kernel_operator.h>`

> 原 OKF v1 卡号：**EC-8**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: unknown type name 'GM_ADDR'
  error: unknown type name '__gm__'
  error: use of undeclared identifier 'atomicAdd'
  error: unknown type name 'bfloat16_t'
  error: no member named 'VF_CALL' in namespace 'AscendC::Simt'
  ```
  (cascade of errors — types, macros, and functions all undefined)
- **Root cause**: `kernel_operator.h` is the master header for AscendC. It pulls in all CANN types (`GM_ADDR`, `__gm__`, `bfloat16_t`, `half`), SIMT APIs (`Simt::VF_CALL`, `Simt::Dim3`), SIMD APIs (`DataCopy`, `Cast`), atomics (`atomicAdd`), and platform macros (`LAUNCH_BOUND`, `__aicore__`). Without it, nothing AscendC-specific compiles.
- **Fix**:
  ```cpp
  // BEFORE (cascade of errors):
  #include <cstdint>
  // missing kernel_operator.h

  // AFTER (compiles):
  #include <kernel_operator.h>    // ✅ MUST be first AscendC include
  #include <cstdint>
  ```
- **Rule**: Every `.h` and `.cpp` file that uses any AscendC type or API must include `<kernel_operator.h>` as its first AscendC include. Standard library headers (`<cstdint>`, `<cstring>`) can come before or after.
- **Related**: None (basic AscendC requirement)

<!-- 迁移自 porter kb/target/ascendc/（EC-8，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
