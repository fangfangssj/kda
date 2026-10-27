---
type: API
title: asc_get_l0c2gm_unitflag
description: 头文件路径为："c_api/cube_datamove/cube_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_store_aux_config/asc_get_l0c2gm_unitflag.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_get_l0c2gm_unitflag

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

数据搬运过程中进行随路量化时，通过调用该接口获取unit_flag设置。

该状态由[asc_set_l0c_copy_config](asc_set_l0c_copy_config.md)配置，其`is_clean_unit_flag`参数的含义和配置覆盖行为参见对应接口。UnitFlag与计算、搬出指令的配合见[UnitFlag特性说明](../../cube_compute/mmad_compute_key_features/unit_flag.md)。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_unitflag()
```

## 参数说明

无

## 返回值说明

unit_flag设置。unit_flag是一种矩阵计算指令和矩阵搬运指令细粒度的并行，开启该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t unitflag_value = asc_get_l0c2gm_unitflag();
```
