---
type: Runbook
title: "`SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)"
description: "`SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)"
tags: [task.debugging, topic.data_movement, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-29-sortconfig-device-side-is-2-field-not-4-field-host.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-29", "SortConfig"]
---

# `SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)

> 原 OKF v1 卡号：**EC-29**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel compile error excess elements in struct initializer on line declaring constexpr SortConfig SORT_CFG = {...}.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Kernel compile error `excess elements in struct initializer` on line declaring `constexpr SortConfig SORT_CFG = {...}`.
- **Root cause**: The hardware `Sort` API has **two** `SortConfig` definitions:
  - **Host-side tiling header** `sort_tiling_intf.h`: 4 fields `{type, isDescend, hasSrcIndex, hasDstIndex}`.
  - **Device-side impl** `sort_impl.h`: 2 fields `{type, isDescend}` — source/destination index options are fixed by overload choice, not config field.
- **Fix**: For device-side (the kernel), use 2-field initializer only: `constexpr SortConfig CFG = {SortType::RADIX_SORT, true};`. Use the simpler Sort overload `Sort<T, isReuse, cfg>(dst, dstIdx, src, tmp, count)` which auto-assigns default indices.
- **Detection**: First-line-of-kernel compile error, no other errors preceding.
- **Evidence**: 9_TopKTopP V2 iter 2 Phase C (2026-04-17). 4-field initializer copied from CANN host-side sample → device compile fails.

<!-- 迁移自 porter kb/target/ascendc/（EC-29，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
