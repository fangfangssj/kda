---
type: API
title: asc_copy_gm2l0b_sync（废弃）
description: 头文件路径为："c_api/composite/cube_datamove_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_copy_gm2l0b_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/cube_datamove_composite.h
---

# asc_copy_gm2l0b_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/cube_datamove_composite.h"`。

**`asc_copy_gm2l0b_sync`接口已废弃，请使用[对应的非同步接口](../cube_datamove/cube_compute_load/asc_copy_gm2l0b.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将矩阵数据从Global Memory搬运到L0B Buffer。

## 函数原型

- 同步搬运

    ```cpp
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int4b_t* dst, __gm__ int4b_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    __aicore__ inline void asc_copy_gm2l0b_sync(__cb__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
    ```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| base_idx | 输入 | 以16*16个数对矩阵进行分块，搬运的起始分块ID。 |
| repeat | 输入 | 迭代次数。 |
| src_stride |输入| 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。 |
| dst_gap | 输入 | 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。 |

## 返回值说明

无

## 流水类型

PIPE_MTE2

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。

## 调用示例

```cpp
//搬运的起始分块为1
constexpr uint16_t base_idx = 1;
//搬运的迭代次数为2
constexpr uint8_t repeat = 2;
//输入的搬运步长为0字节，输出的搬运步长为1024字节
constexpr uint16_t src_stride = 0;
constexpr uint16_t dst_gap = 1;
// src表示源操作数的起始地址
__cb__ half dst[256];
asc_copy_gm2l0b_sync(dst, src, base_idx, repeat, src_stride, dst_gap);
```
