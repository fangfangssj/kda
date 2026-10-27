---
type: Runbook
title: "pybind path CAN use multiple MTE2 TQue — agent must not blame pybind structure"
description: "applies_to: soc=Ascend950PR (V351/arch35); cann=9.0.0; paradigm=pybind (build_ascendc.py / aclruntime CMake)"
tags: [task.debugging, topic.implementation.masking, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-266-pybind-path-can-use-multiple-mte2-tque-agent-must.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-266"]
---

# pybind path CAN use multiple MTE2 TQue — agent must not blame pybind structure

> 原 OKF v1 卡号：**OL-266**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when a worker claims "NPUKernelBench pybind path doesn't support multiple VECIN/VECOUT TQue" and uses this as justification to abandon a correct TQue-based desi

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.0.0; paradigm=pybind (build_ascendc.py / aclruntime CMake)`
`verified_on: soc=Ascend950PR; cann=9.0.0`

- **The fact**: NPUKernelBench pybind build path fully supports multiple VECIN/VECOUT TQue on MTE2. When a kernel using two MTE2 TQue fails, the pybind structure is NOT the root cause.
- **Counter-example**: `workspace/group_norm_silu/kernel/group_norm_silu_kernels.cpp` — **6 TQue** (3 VECIN + 3 VECOUT) working correctly on A5 via the same pybind build path. This proves it is possible.
- **Incident**: masked_select_v3 kw-1 (2026-06-25) — worker used two VECIN TQue (data + mask), hit error 95, and concluded "pybind path cannot support multiple VECIN TQue." This conclusion was wrong. kw-2 fixed the issue without changing the build path. The real cause may have been NPU0 hardware state at the time, not a structural pybind limitation.
- **Trigger**: when a worker claims "NPUKernelBench pybind path doesn't support multiple VECIN/VECOUT TQue" and uses this as justification to abandon a correct TQue-based design or fall back to PipeBarrier workarounds — STOP. The claim is false. Point the worker at the group_norm_silu counter-example.
- **Related**: OL-4

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-266（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
