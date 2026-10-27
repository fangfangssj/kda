---
type: Runbook
title: "`Simt::atomicAdd` — wrong namespace"
description: "`Simt::atomicAdd` — wrong namespace"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.implementation.atomic, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-7-simt-atomicadd-wrong-namespace.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-7", "Simt::atomicAdd"]
---

# `Simt::atomicAdd` — wrong namespace

> 原 OKF v1 卡号：**EC-7**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: no member named 'atomicAdd' in namespace 'AscendC::Simt'
  ```
  or:
  ```
  error: call to 'atomicAdd' is ambiguous
  ```
  (when both `Simt::atomicAdd` and global `atomicAdd` are attempted)
- **Root cause**: `atomicAdd` on AscendC is a **global built-in function**, not a member of the `AscendC::Simt` namespace. This differs from other Simt APIs like `Simt::VF_CALL`, `Simt::Dim3`, `Simt::WarpReduceAddSync` which are namespaced.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  Simt::atomicAdd(base + offset, value);       // ❌ not in Simt namespace
  AscendC::Simt::atomicAdd(base + offset, value);  // ❌ same error

  // AFTER (compiles):
  atomicAdd(base + offset, value);             // ✅ global built-in, no namespace
  ```
- **Supported types**: `float`, `half`, `bfloat16_t`, `int32_t` — all use the same unqualified `atomicAdd`.

<!-- 迁移自 porter kb/target/ascendc/（EC-7，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
