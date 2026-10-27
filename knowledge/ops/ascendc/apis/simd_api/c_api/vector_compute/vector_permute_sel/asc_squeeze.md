---
type: API
title: asc_squeeze
description: 通过比较掩码src1，将src0中的向量压缩成更短的向量，存储到dst中。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_permute_sel/asc_squeeze.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_permute_sel.h
---

# asc_squeeze

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_permute_sel.h"`。

通过比较掩码src1，将src0中的向量压缩成更短的向量，存储到dst中。

## 函数原型

  ```cpp
  __aicore__ inline void asc_squeeze(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
  __aicore__ inline void asc_squeeze(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, uint8_t repeat, uint8_t dst_block_stride, uint8_t src0_block_stride, uint8_t src1_block_stride, uint8_t dst_repeat_stride, uint8_t src0_repeat_stride, uint8_t src1_repeat_stride)
  ```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述                                |
|:----------| :-----|:----------------------------------|
| dst       | 输出 | 目的操作数（矢量）的起始地址。                   |
| src0、src1 | 输入 | 源操作数（矢量）的起始地址。                        |
| repeat | 输入 | 迭代次数。                       |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| src0_block_stride | 输入 | 源操作数0单次迭代内不同DataBlock间地址步长。 |
| src1_block_stride | 输入 | 源操作数1单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| src0_repeat_stride | 输入 | 源操作数0相邻迭代间相同DataBlock的地址步长。 |
| src1_repeat_stride | 输入 | 源操作数1相邻迭代间相同DataBlock的地址步长。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src0、src1的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

```cpp
// total_length指参与计算的数据总长度
constexpr uint64_t total_length = 128;
__ubuf__ uint16_t src0[total_length];
__ubuf__ uint16_t src1[total_length];
__ubuf__ uint16_t dst[total_length];
uint8_t repeat = 1;
uint8_t dst_block_stride = 1;
uint8_t src0_block_stride = 1;
uint8_t src1_block_stride = 1;
uint8_t dst_repeat_stride = 8;
uint8_t src0_repeat_stride = 8;
uint8_t src1_repeat_stride = 8;
asc_squeeze(dst, src0, src1, repeat, dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride, src1_repeat_stride);
```
