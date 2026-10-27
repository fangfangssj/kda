---
type: API
title: asc_create_iter_reg (废弃)
description: 头文件路径为："c_api/reg_compute/compute/reg_addr_reg.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_create_iter_reg_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_addr_reg.h
---

# asc_create_iter_reg (废弃)

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_addr_reg.h"`。

**该接口已废弃，请使用[asc_update_addr_reg](../reg_compute/reg_addr_reg/asc_update_addr_reg.md)实现此功能。**

地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量，该重载接口同步支持1-4个offset参数，并支持初始化b8/b16/b32位宽的寄存器，适配1-4维Tensor。底层在多维循环中自动计算地址偏移，消除标量地址运算开销。

## 函数原型

  ```cpp
  __simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset0, uint32_t offset1)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset0, uint32_t offset1)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset0, uint32_t offset1)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset0, uint32_t offset1, uint32_t offset2)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset0, uint32_t offset1, uint32_t offset2)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset0, uint32_t offset1, uint32_t offset2)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)
  __simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)
  ```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| offset       | 输入 | 地址偏移量。 |
| offset0      | 输入 | 地址偏移量。 |
| offset1     | 输入 | 地址偏移量。 |
| offset2     | 输入 | 地址偏移量。 |
| offset3     | 输入 | 地址偏移量。 |

## 返回值说明

地址寄存器。

## 流水类型

PIPE_V

## 约束说明

- 请勿在循环内使用 if/else。
- iter_reg必须在最内层循环体内赋值使用。
- 不同步长的操作数，建议使用多个iter_reg，避免地址被覆盖。

## 调用示例

见[地址寄存器调用示例](../defs/type/data_type_definition.md#example4)。

- [asc_create_iter_reg (废弃)：固定 reference 调用约束](../../../../guides/regbase/asc_create_iter_reg_deprecated_reference_contract.md)
