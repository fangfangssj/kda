---
type: API
title: asc_datablock_reduce_max
description: 执行数据块内的求最大值规约（Reduce Max）操作。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_reduce/asc_datablock_reduce_max.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_reduce.h
---

# asc_datablock_reduce_max

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_reduce.h"`。

执行数据块内的求最大值规约（Reduce Max）操作。

## 函数原型

- 前n个数据连续计算

    ```cpp
    __aicore__ inline void asc_datablock_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_datablock_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_datablock_reduce_max(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    __aicore__ inline void asc_datablock_reduce_max(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dst_repeat_stride, uint16_t src_block_stride, uint16_t src_repeat_stride)
    ```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| dst     | 输出     | 目的操作数（矢量）的起始地址。 |
| src     | 输入     | 源操作数（矢量）的起始地址。 |
| count   | 输入     | 参与连续计算的元素个数。|
| repeat | 输入 | 迭代次数。|
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。<br>输入类型位宽为16bit时，单位为16Byte，输入类型位宽为32bit时，单位为32Byte。|
| src_block_stride | 输入 | 源操作数单次迭代内不同DataBlock间地址步长。|
| src_repeat_stride | 输入 | 源操作数相邻迭代间相同DataBlock的地址步长。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint32_t src_length = 256;
constexpr uint32_t dst_length = 16;
__ubuf__ half src[src_length];
__ubuf__ half dst[dst_length];

// 使用高维切分计算接口，需设置mask
asc_set_mask_norm();
asc_set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);    // 设置每个迭代内所有元素均参与计算。

// 每次repeat256B，2次repeat，无间隔
asc_datablock_reduce_max(dst, src, 2, 1, 1, 8);
```

结果示例：

```
输入数据src：[1 2 3 ... 16 17 ... 32 ... 225... 256]
输出数据dst：[16 32 ... 256]
```
