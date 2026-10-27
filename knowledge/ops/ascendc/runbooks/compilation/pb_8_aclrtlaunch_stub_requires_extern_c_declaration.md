---
type: Runbook
title: "aclrtlaunch Stub Requires extern \"C\" Declaration"
description: "aclrtlaunch Stub Requires extern \"C\" Declaration"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-8-aclrtlaunch-stub-requires-extern-c-declaration.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["PB-8"]
---

# aclrtlaunch Stub Requires extern "C" Declaration

> 原 OKF v1 卡号：**PB-8**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Linker error undefined reference to aclrtlaunch_xxx(...) when calling kernel from test code

- **Symptom**: Linker error `undefined reference to aclrtlaunch_xxx(...)` when calling kernel from test code
- **Root cause**: Auto-generated `host_stub.cpp` exports functions as C symbols (no name mangling). Test code declaring them as C++ gets mangled names → linker mismatch.
- **Workaround**: Always use `extern "C" { uint32_t aclrtlaunch_xxx(...); }` in test code
- **Status**: By design (not a bug, but easy to forget)
- **Evidence**: MXFP4 test (2026-04-07)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Operational Issues

<!-- 迁移自 porter kb/target/ascendc/（PB-8，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
