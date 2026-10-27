---
type: Runbook
title: "Auto-generated host_stub.cpp `FreeAscendMemDevice` races with async kernel execution on V220 [V220]"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-38-auto-generated-host-stub-cpp-freeascendmemdevice-r.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3]
aliases: ["PB-38", "FreeAscendMemDevice"]
---

# Auto-generated host_stub.cpp `FreeAscendMemDevice` races with async kernel execution on V220 [V220]

> 原 OKF v1 卡号：**PB-38**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - First kernel launch produces bit-exact output; subsequent launches show max_abs_diff 5-7 for fp32 tensors with random values.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910_9382 (V220); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 V351 — LaunchAscendKernel may be synchronous on V351)`

- **Severity**: HIGH (silent data corruption on multi-launch)
- **Status**: CONFIRMED 2026-05-21 12_Permute a3-ds kw-3
- **Symptom**: First kernel launch produces bit-exact output; subsequent launches show max_abs_diff 5-7 for fp32 tensors with random values.
- **Root cause**: Auto-generated `host_stub.cpp` calls `FreeAscendMemDevice(overflow_buf)` immediately after `LaunchAscendKernel`, but kernel executes asynchronously. On subsequent launches the same physical memory may be reused while previous kernel still references it.
- **Fix**: Call `torch.npu.synchronize()` after each kernel launch in pybind wrapper.
- **Cross-ref**: OL-66 (torch::zeros not stream-ordered — same class of stream-safety issue).

<!-- 迁移自 porter kb/target/ascendc/（PB-38，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
