---
type: Runbook
title: "aclrtlaunch Undefined Reference (Linker)"
description: "aclrtlaunch Undefined Reference (Linker)"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-10-aclrtlaunch-undefined-reference-linker.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-10"]
---

# aclrtlaunch Undefined Reference (Linker)

> 原 OKF v1 卡号：**EC-10**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - undefined reference to 'aclrtlaunch_xxx(...)'

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: `undefined reference to 'aclrtlaunch_xxx(...)'`
- **Root cause**: Auto-generated `host_stub.cpp` exports kernel launch functions as **C symbols** (no mangling). Test code that declares them without `extern "C"` gets C++ mangled names → linker mismatch.
- **Fix**:
  ```cpp
  // ❌ Wrong — C++ mangling
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);

  // ✅ Correct — C linkage
  extern "C" {
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);
  }
  ```
- **Related**: PB-8 in PLATFORM_BUGS.md

<!-- 迁移自 porter kb/target/ascendc/（EC-10，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
