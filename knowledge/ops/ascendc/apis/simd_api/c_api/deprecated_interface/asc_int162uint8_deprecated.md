---
type: API
title: asc_int162uint8（废弃）
description: 该接口通过接口后缀来控制目的操作数写入位置（仅包括源操作数、目的操作数、掩码寄存器三个参数）的原型已废弃，请使用asc_int162uint8的显式位置参数重载替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_int162uint8_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/reg_convert.h
---

# asc_int162uint8（废弃）

## 功能说明

头文件路径为：`"c_api/reg_compute/reg_convert.h"`。

**该接口通过接口后缀来控制目的操作数写入位置（仅包括源操作数、目的操作数、掩码寄存器三个参数）的原型已废弃，请使用[asc_int162uint8](../reg_compute/reg_convert/asc_int162uint8.md)的显式位置参数重载替代。**

将src中int16_t类型的元素转换为uint8_t类型，并将结果写入dst。支持非饱和与饱和两种模式。

由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选择数据写入索引为奇数的位置或偶数的位置。

## 函数原型

```cpp
// 非饱和模式，结果写入dst各组内第0个位置，直接截断。mask每2bit为一组，仅组内LSB为1时选取对应位置的int16_t元素。
__simd_callee__ inline void asc_int162uint8(vector_uint8_t& dst, vector_int16_t src, vector_bool mask)
// 非饱和模式，结果写入dst各组内第1个位置，直接截断。mask每2bit为一组，仅组内LSB为1时选取对应位置的int16_t元素。
__simd_callee__ inline void asc_int162uint8_v2(vector_uint8_t& dst, vector_int16_t src, vector_bool mask)
// 饱和模式，超出[0, 255]范围的值钳位到边界。结果写入dst各组内第0个位置。mask每2bit为一组，仅组内LSB为1时选取对应位置的int16_t元素。
__simd_callee__ inline void asc_int162uint8_sat(vector_uint8_t& dst, vector_int16_t src, vector_bool mask)
// 饱和模式，超出[0, 255]范围的值钳位到边界。结果写入dst各组内第1个位置。mask每2bit为一组，仅组内LSB为1时选取对应位置的int16_t元素。
__simd_callee__ inline void asc_int162uint8_sat_v2(vector_uint8_t& dst, vector_int16_t src, vector_bool mask)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../defs/type/data_type_definition.md)。

## 返回值说明

无

## 约束说明

mask控制源操作数是否参与计算，源操作数不参与计算的元素在输出对应位置置零。

## 调用示例

```cpp
__simd_vf__ inline void int162uint8_vf(__ubuf__ uint8_t* dst_addr, __ubuf__ int16_t* src_addr, uint32_t count, uint16_t one_repeat_size, uint16_t repeat_time)
{
    vector_uint8_t dst;
    vector_int16_t src;
    vector_bool mask;
    for (uint16_t i = 0; i < repeat_time; ++i) {
        mask = asc_update_mask_b16(count);
        asc_loadalign(src, src_addr + i * one_repeat_size);
        asc_int162uint8(dst, src, mask);
        asc_storealign(dst_addr + i * one_repeat_size, dst, mask);
    }
}
```
