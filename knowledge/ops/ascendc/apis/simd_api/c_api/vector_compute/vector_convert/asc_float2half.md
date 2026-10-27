---
type: API
title: asc_float2half
description: 将float类型数据转换为half类型，支持多种舍入模式：
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_convert/asc_float2half.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_convert.h
---

# asc_float2half

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_convert.h"`。

将float类型数据转换为half类型，支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入。
- ROUND舍入模式：四舍五入舍入。
- FLOOR舍入模式：向负无穷舍入。
- CEIL舍入模式：向正无穷舍入。
- TRUNC舍入模式：向零舍入。
- ODD舍入模式：Von Neumann rounding，最近邻奇数舍入。

## 函数原型

- 前n个数据计算

  ```cpp
  // 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // RINT舍入模式
  __aicore__ inline void asc_float2half_rn(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2half_rna(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2half_rd(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2half_ru(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2half_rz(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  // ODD舍入模式
  __aicore__ inline void asc_float2half_ro(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  ```

- 高维切分计算

  ```cpp
  // 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // RINT舍入模式
  __aicore__ inline void asc_float2half_rn(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // ROUND舍入模式
  __aicore__ inline void asc_float2half_rna(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2half_rd(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // CEIL舍入模式
  __aicore__ inline void asc_float2half_ru(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2half_rz(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  // ODD舍入模式
  __aicore__ inline void asc_float2half_ro(__ubuf__ half* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_block_stride, uint16_t src_block_stride, uint16_t dst_repeat_stride, uint16_t src_repeat_stride)
  ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src  | 输入 | 源操作数（矢量）的起始地址。|
| count | 输入 | 参与计算的元素个数。 |
| repeat |输入 | 迭代次数。 |
| dst_block_stride |输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src_block_stride |输入 | 源操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride |输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src_repeat_stride |输入 | 源操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

```cpp
constexpr uint32_t total_length = 256;
__ubuf__ float src[total_length];
__ubuf__ half dst[total_length];
asc_float2half(dst, src, total_length);
```
