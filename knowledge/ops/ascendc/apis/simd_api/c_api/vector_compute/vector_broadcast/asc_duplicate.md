---
type: API
title: asc_duplicate
description: 执行矢量复制（Duplicate）操作，将标量值复制填充到矢量中。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_broadcast/asc_duplicate.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_broadcast.h
---

# asc_duplicate

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_broadcast.h"`。

执行矢量复制（Duplicate）操作，将标量值复制填充到矢量中。

## 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint8_t repeat, uint16_t dst_block_stride, uint16_t dst_repeat_stride)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（标量）。 |
| count | 输入 | 参与连续复制的元素个数。 |
| repeat | 输入  | 迭代次数。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint32_t dst_length = 128;
__ubuf__ uint16_t dst[dst_length];
uint16_t val = 18;
asc_duplicate(dst, val, dst_length);
```

结果示例：

```
输出数据dst：[18 18 18 ... 18]
```
