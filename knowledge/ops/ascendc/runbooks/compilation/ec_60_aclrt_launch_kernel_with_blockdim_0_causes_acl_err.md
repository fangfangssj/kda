---
type: Runbook
title: "ACLRT_LAUNCH_KERNEL with blockDim=0 causes ACL_ERROR_RT_PARAM_INVALID (107000) on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-60-aclrt-launch-kernel-with-blockdim-0-causes-acl-err.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["EC-60"]
---

# ACLRT_LAUNCH_KERNEL with blockDim=0 causes ACL_ERROR_RT_PARAM_INVALID (107000) on V220

> 原 OKF v1 卡号：**EC-60**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel compiles cleanly (static_check 10/10 PASS), but every launch fails with ACL_ERROR_RT_PARAM_INVALID (107000).

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all`

- **Severity**: CRITICAL (kernel never launches, but compilation succeeds silently)
- **Status**: CONFIRMED 2026-05-22 26_AvgPool3d a3-ds
- **Symptom**: Kernel compiles cleanly (static_check 10/10 PASS), but every launch fails with `ACL_ERROR_RT_PARAM_INVALID (107000)`.
- **Root cause**: `ACLRT_LAUNCH_KERNEL(kernel_name)(0, stream, args...)` passes blockDim=0. ACL runtime rejects it; kernel's `GetBlockNum()` returns 0, causing division-by-zero in per-block work distribution.
- **Fix**: Replace `ACLRT_LAUNCH_KERNEL` with explicit `extern "C"` declarations + dynamic nblk computation (floor at 1, cap at 56 for V220).
- **Detection**: grep for `ACLRT_LAUNCH_KERNEL.*\(0,` in pybind11.cpp.
- **Evidence**: 26_AvgPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: PB-28 (KERNEL_TASK_TYPE_DEFAULT is arch35-only — also produces 107000 on V220).

<!-- 迁移自 porter kb/target/ascendc/（EC-60，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
