---
type: API
title: asc_get_l0c2gm_prequant
description: 头文件路径为："c_api/cube_datamove/cube_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_store_aux_config/asc_get_l0c2gm_prequant.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_get_l0c2gm_prequant

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

数据搬运过程中进行随路量化时，通过调用该接口获取量化操作前矢量的起始地址。

量化模式、参数编码和准备流程见[随路量化](../cube_store_key_features/accompanying_quantization.md)；量化与激活参数的配合见[随路量化与随路ReLU场景组合](../cube_store_key_features/accompanying_quantization_and_relu_scenario_combination.md)。

该状态由[asc_set_l0c_copy_config](asc_set_l0c_copy_config.md)配置，其`quant_pre_addr`参数的含义和配置覆盖行为参见对应接口。返回值是FPC寄存器中的配置字段，不是量化或激活计算后的数据。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_prequant()
```

## 参数说明

无

## 返回值说明

量化操作前矢量的起始地址。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t prequant_addr = asc_get_l0c2gm_prequant();
```
