---
type: API
title: asc_set_l13d_padding（废弃）
description: asc_set_l13d_padding接口已废弃。请使用asc_set_l12l0a_3d_padding接口替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_l13d_padding_deprecated.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_set_l13d_padding（废弃）

## 功能说明

**`asc_set_l13d_padding`接口已废弃。请使用[asc_set_l12l0a_3d_padding](../cube_datamove/cube_load_aux_config/asc_set_l12l0a_3d_padding.md)接口替代。**

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

设置Pad属性描述，用于在调用[asc_copy_l12l0a](../cube_datamove/cube_compute_load/index.md)接口时配置填充数值。

## 函数原型

```cpp
__aicore__ inline void asc_set_l13d_padding(uint64_t config)
__aicore__ inline void asc_set_l13d_padding(half config)
__aicore__ inline void asc_set_l13d_padding(int16_t config)
__aicore__ inline void asc_set_l13d_padding(uint16_t config)
```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| config | 输入     | Pad填充值的数值。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
constexpr uint64_t config = 0;
asc_set_l13d_padding(config);
```
