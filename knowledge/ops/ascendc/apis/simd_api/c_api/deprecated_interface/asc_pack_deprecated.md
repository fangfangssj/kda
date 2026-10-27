---
type: API
title: asc_pack (废弃)
description: 头文件路径为："c_api/reg_compute/compute/reg_permute_sel.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_pack_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_permute_sel.h
---

# asc_pack (废弃)

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_permute_sel.h"`。

**该接口已废弃，请使用 [asc_pack_to_low](../reg_compute/reg_permute_sel/asc_pack_to_low.md) 和 [asc_pack_to_high](../reg_compute/reg_permute_sel/asc_pack_to_high.md) 替代。**

将源操作数中的元素选取低8位（b16）、低16位（b32）、低32位（b64）写入目的操作数的低半部分或高半部分。

- asc_pack：将源操作数写入目的操作数的低半部分。

- asc_pack_v2：将源操作数写入目的操作数的高半部分。

## 函数原型

```cpp
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_uint16_t src)
__simd_callee__ inline void asc_pack(vector_uint8_t& dst, vector_int16_t src)
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_uint32_t src)
__simd_callee__ inline void asc_pack(vector_uint16_t& dst, vector_int32_t src)
__simd_callee__ inline void asc_pack(vector_bool& dst, vector_bool src)
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_uint16_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint8_t& dst, vector_int16_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_uint32_t src)
__simd_callee__ inline void asc_pack_v2(vector_uint16_t& dst, vector_int32_t src)
__simd_callee__ inline void asc_pack_v2(vector_bool& dst, vector_bool src)
```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量数据寄存器/掩码寄存器）。            |
| src | 输入    | 源操作数（矢量数据寄存器/掩码寄存器）。            |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../defs/type/data_type_definition.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_uint8_t dst;
vector_uint16_t src;
asc_pack(dst, src);    // 将src的低8位写入dst的低半部分
asc_pack_v2(dst, src);    // 将src的低8位写入dst的高半部分
```

- [asc_pack (废弃)：固定 reference 调用约束](../../../../guides/regbase/asc_pack_deprecated_reference_contract.md)
