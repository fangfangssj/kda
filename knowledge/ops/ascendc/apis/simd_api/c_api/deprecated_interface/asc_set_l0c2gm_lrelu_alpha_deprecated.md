---
type: API
title: asc_set_l0c2gm_lrelu_alpha（废弃）
description: asc_set_l0c2gm_lrelu_alpha接口已废弃。请使用asc_set_l0c_copy_lrelu_alpha接口替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_l0c2gm_lrelu_alpha_deprecated.md
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

# asc_set_l0c2gm_lrelu_alpha（废弃）

## 功能说明

**`asc_set_l0c2gm_lrelu_alpha`接口已废弃。请使用[asc_set_l0c_copy_lrelu_alpha](../cube_datamove/cube_store_aux_config/asc_set_l0c_copy_lrelu_alpha.md)接口替代。**

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

用于设置asc_copy_l0c2l1或asc_copy_l0c2gm接口计算过程中使用的Leaky ReLU alpha值。该值只支持half和float两种数据类型。

## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_lrelu_alpha(half& config)
__aicore__ inline void asc_set_l0c2gm_lrelu_alpha(float& config)
```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | Leaky ReLU alpha值。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
half config = 0.5;
asc_set_l0c2gm_lrelu_alpha(config);
```
