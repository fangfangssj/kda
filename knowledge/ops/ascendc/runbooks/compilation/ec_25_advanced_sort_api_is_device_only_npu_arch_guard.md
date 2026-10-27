---
type: Runbook
title: "Advanced Sort API is device-only (__NPU_ARCH__ guard)"
description: "Advanced Sort API is device-only (__NPU_ARCH__ guard)"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-25-advanced-sort-api-is-device-only-npu-arch-guard.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-25"]
---

# Advanced Sort API is device-only (__NPU_ARCH__ guard)

> 原 OKF v1 卡号：**EC-25**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Sort<T, isReuse, config>() compilation fails with "SortConfig not found" or "SortType not found".

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: `Sort<T, isReuse, config>()` compilation fails with "SortConfig not found" or "SortType not found".
- **Root cause**: The advanced Sort API (`adv_api/sort/sort.h`) is only available when `__NPU_ARCH__` is defined during device compilation. Host build stubs don't include these declarations.
- **Fix**: Guard Sort API usage with `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ > 0)`. In host compilation, the function can be a stub or not compiled.
- **Evidence**: Sort kernel development (2026-04-14).

<!-- 迁移自 porter kb/target/ascendc/（EC-25，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
