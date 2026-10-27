---
type: Runbook
title: "Tiling CPU→NPU copy must happen AFTER all fields finalized"
description: "Tiling CPU→NPU copy must happen AFTER all fields finalized"
tags: [task.debugging, topic.data_movement, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-20-tiling-cpu-npu-copy-must-happen-after-all-fields-f.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-20"]
---

# Tiling CPU→NPU copy must happen AFTER all fields finalized

> 原 OKF v1 卡号：**EC-20**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Wrong results — tiling field has stale value on NPU

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Wrong results — tiling field has stale value on NPU
- **Root cause**: If pybind writes `tiling.field = X` after `tiling_npu = tiling_cpu.to(device)`, the NPU copy has old value
- **Fix**: Finalize ALL tiling fields, then copy once
- **Evidence**: Pad V2 mode routing bug (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-20，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
