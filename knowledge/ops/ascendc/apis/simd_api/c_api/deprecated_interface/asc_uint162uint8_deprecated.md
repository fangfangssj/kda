---
type: API
title: asc_uint162uint8（废弃）
description: 该接口通过接口后缀来控制目的操作数写入位置（仅包括源操作数、目的操作数、掩码寄存器三个参数）的原型已废弃，请使用asc_uint162uint8的显式位置参数重载替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_uint162uint8_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/reg_convert.h
---

# asc_uint162uint8（废弃）

## 功能说明

头文件路径为：`"c_api/reg_compute/reg_convert.h"`。

**该接口通过接口后缀来控制目的操作数写入位置（仅包括源操作数、目的操作数、掩码寄存器三个参数）的原型已废弃，请使用[asc_uint162uint8](../reg_compute/reg_convert/asc_uint162uint8.md)的显式位置参数重载替代。**

将uint16_t类型数据转化为uint8_t类型，无舍入模式，支持饱和/非饱和模式。

关于饱和/非饱和模式的详细说明，请参见[舍入模式与饱和模式](../reg_compute/reg_convert/rounding_mode.md)。

由于源操作数与目的操作数类型位宽比为2:1，写入数据时需要将一个VL大小的数据分为两部分，根据不同接口选择数据写入索引为奇数的位置或偶数的位置。

## 函数原型

```cpp
// 非饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_uint162uint8(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 饱和模式，数据写入索引为偶数的位置
__simd_callee__ inline void asc_uint162uint8_sat(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 非饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_uint162uint8_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
// 饱和模式，数据写入索引为奇数的位置
__simd_callee__ inline void asc_uint162uint8_sat_v2(vector_uint8_t& dst, vector_uint16_t src, vector_bool mask)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数（矢量数据寄存器）。|
| src | 输入 | 源操作数（矢量数据寄存器）。|
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../defs/type/data_type_definition.md)。

## 返回值说明

无

## 约束说明

- mask控制源操作数是否参与计算，源操作数不参与计算的元素在输出对应位置置零。
- 当数据写入目的操作数索引为奇数的位置，索引为偶数位置的数值置零。反之亦然。
- 开启饱和模式和非饱和模式时，需配置ctrl寄存器，ctrl寄存器的详细说明请参见[asc_set_ctrl.md](../spr/asc_set_ctrl.md)。

## 调用示例

```cpp
vector_uint16_t src;
vector_uint8_t dst;
vector_bool mask = asc_create_mask_b16(PAT_ALL);
asc_loadalign(src, src_addr); // src_addr是外部输入的Unified Buffer（UB）内存空间地址。
asc_uint162uint8(dst, src, mask);
```
