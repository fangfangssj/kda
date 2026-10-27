---
type: API
title: asc_get_scalar_load_cache_mode
description: 读取当前标量读（load）路径的L2 Cache管理策略，并返回asc_load_l2_cache_mode枚举值。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_get_scalar_load_cache_mode.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/cache_ctrl/cache_ctrl.h
---

# asc_get_scalar_load_cache_mode

## 功能说明

头文件路径为：`"c_api/cache_ctrl/cache_ctrl.h"`。

读取当前标量读（load）路径的L2 Cache管理策略，并返回[asc_load_l2_cache_mode](../defs/enum/asc_load_l2_cache_mode.md)枚举值。该值可通过[asc_set_scalar_cache_mode](asc_set_scalar_cache_mode.md)接口设置，关于策略的详细说明请参考该接口。

常用于临时切换策略前保存原配置，或在设置后做回读校验。

## 函数原型

```c
__aicore__ inline asc_load_l2_cache_mode asc_get_scalar_load_cache_mode()
```

## 参数说明

无

## 返回值说明

当前标量load路径的L2 Cache管理策略，类型为[asc_load_l2_cache_mode](../defs/enum/asc_load_l2_cache_mode.md)。

## 流水类型

PIPE_S

## 约束说明

本接口仅反映标量load侧策略，与标量store侧以及向量搬运接口参数中的`l2_cache_mode`无关。

## 调用示例

本接口用于保存标量load路径当前策略，完整调用示例请参考[asc_set_scalar_cache_mode的场景A](asc_set_scalar_cache_mode.md#asc_set_scalar_cache_mode_scenario_a)。
