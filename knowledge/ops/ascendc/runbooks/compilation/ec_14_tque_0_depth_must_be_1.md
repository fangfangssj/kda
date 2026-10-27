---
type: Runbook
title: "`TQue<..., 0>` — depth must be >= 1"
description: "`TQue<..., 0>` — depth must be >= 1"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-14-tque-0-depth-must-be-1.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-14"]
---

# `TQue<..., 0>` — depth must be >= 1

> 原 OKF v1 卡号：**EC-14**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: static assertion failed: must use AllocTensor<LocalTensor&> api while tque's depth is zero
  ```
- **Root cause**: `TQue` template's second parameter is the depth (number of buffer slots). Depth 0 means "use pass-by-reference AllocTensor API" which has a completely different usage pattern. Standard AllocTensor/EnQue/DeQue/FreeTensor requires depth >= 1.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::TQue<AscendC::TPosition::VECIN, 0> xQueue_;

  // AFTER (works):
  AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
  ```
- **Evidence**: Cumsum V1 build failure (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-14，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
