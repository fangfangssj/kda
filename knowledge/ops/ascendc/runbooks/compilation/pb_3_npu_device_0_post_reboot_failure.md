---
type: Runbook
title: "NPU Device 0 Post-Reboot Failure"
description: "NPU Device 0 Post-Reboot Failure"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-3-npu-device-0-post-reboot-failure.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["PB-3"]
---

# NPU Device 0 Post-Reboot Failure

> 原 OKF v1 卡号：**PB-3**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Error 507033 on device 0 after server reboot (2026-04-01)

- **Symptom**: Error 507033 on device 0 after server reboot (2026-04-01)
- **Affected**: A5 server 198.51.100.35, device 0 only
- **Workaround**: Use devices 1-4, 7
- **Status**: Hardware issue, may require RMA

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Bisheng Compiler Bugs

<!-- 迁移自 porter kb/target/ascendc/（PB-3，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
