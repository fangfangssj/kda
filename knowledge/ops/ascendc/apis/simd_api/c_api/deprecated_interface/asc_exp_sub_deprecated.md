---
type: API
title: asc_exp_sub（废弃）
description: 输入数据类型为half且不带src_pos参数的asc_exp_sub原型以及asc_exp_sub_v2原型已废弃。新接口请参见asc_exp_sub。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_exp_sub_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_fused.h
---

# asc_exp_sub（废弃）

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_fused.h"`。

**输入数据类型为`half`且不带`src_pos`参数的`asc_exp_sub`原型以及`asc_exp_sub_v2`原型已废弃。新接口请参见[asc_exp_sub](../reg_compute/reg_fused/asc_exp_sub.md)。**

这些接口将`src0`与`src1`按元素相减，并计算以差值为指数的自然指数函数，将计算结果写入`dst`。

### 迁移说明

**表1** 新旧原型对应关系

| 旧原型 | 新原型 |
| ------ | ------ |
| `asc_exp_sub(dst, vector_half src0, vector_half src1, mask)` | `asc_exp_sub_half2float(dst, src0, src1, mask, ASC_POSITION_EVEN)` |
| `asc_exp_sub_v2(dst, vector_half src0, vector_half src1, mask)` | `asc_exp_sub_half2float(dst, src0, src1, mask, ASC_POSITION_ODD)` |
| `asc_exp_sub_v2(dst, vector_float src0, vector_float src1, mask)` | `asc_exp_sub(dst, src0, src1, mask)` |

## 函数原型

```c
__simd_callee__ inline void asc_exp_sub(vector_float& dst,
                                        vector_half src0,
                                        vector_half src1,
                                        vector_bool mask)

__simd_callee__ inline void asc_exp_sub_v2(vector_float& dst,
                                           vector_half src0,
                                           vector_half src1,
                                           vector_bool mask)

__simd_callee__ inline void asc_exp_sub_v2(vector_float& dst,
                                           vector_float src0,
                                           vector_float src1,
                                           vector_bool mask)

```

## 参数说明

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| ------ | --------- | ---- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src0 | 输入 | 源操作数0（矢量数据寄存器）。 |
| src1 | 输入 | 源操作数1（矢量数据寄存器）。 |
| mask | 输入 | 掩码寄存器，用于指示参与计算的元素。对应位置为1时参与计算，为0时不参与计算；未参与计算的元素在输出中置0。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../defs/type/data_type_definition.md)。

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
vector_float dst;
vector_half src0;
vector_half src1;
vector_bool mask;

// 旧接口：读取half源操作数的偶数索引元素。
asc_exp_sub(dst, src0, src1, mask);

// 替代接口：显式指定源操作数读取位置。
asc_exp_sub_half2float(dst, src0, src1, mask, ASC_POSITION_EVEN);
```
