---
type: Runbook
title: "-O2 Required for NPU (No -O0)"
description: "-O2 Required for NPU (No -O0)"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-5-o2-required-for-npu-no-o0.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["PB-5"]
---

# -O2 Required for NPU (No -O0)

> 原 OKF v1 卡号：**PB-5**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel may produce wrong results or crash with -O0 on NPU

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Symptom**: Kernel may produce wrong results or crash with -O0 on NPU
- **Affected**: All NPU builds
- **Workaround**: Always use -O2 for NPU builds (-O0 only for CPU debug mode)
- **Status**: By design (bisheng optimizations required for correct codegen)

<!-- 迁移自 porter kb/target/ascendc/（PB-5，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
