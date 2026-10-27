---
type: Runbook
title: "SortConfig must be global constexpr for template NTTP"
description: "SortConfig must be global constexpr for template NTTP"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-24-sortconfig-must-be-global-constexpr-for-template-n.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-24"]
---

# SortConfig must be global constexpr for template NTTP

> 原 OKF v1 卡号：**EC-24**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - constexpr SortConfig cfg = {...} inside a function body → "cannot use as template non-type parameter" compile error.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: `constexpr SortConfig cfg = {...}` inside a function body → "cannot use as template non-type parameter" compile error.
- **Root cause**: C++ template constraint — `const SortConfig&` NTTP requires the variable to have external linkage / global scope.
- **Fix**: Declare `SortConfig` at namespace/global scope, outside any function body.
- **Evidence**: Sort kernel development (2026-04-14).

<!-- 迁移自 porter kb/target/ascendc/（EC-24，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
