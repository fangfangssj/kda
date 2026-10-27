---
type: API
title: asc_max_scalar_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_max_scalar_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_max_scalar_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_max_scalar_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_arith/asc_max_scalar.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

矢量src的逐个元素与标量value比较大小，接着按照对应的比特位将最大值存入dst中。计算公式如下：

$$
dst_i = max(src_i, value)
$$

## 函数原型

- 同步计算

  ```cpp
  __aicore__ inline void asc_max_scalar_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t value, uint32_t count)
  __aicore__ inline void asc_max_scalar_sync(__ubuf__ half* dst, __ubuf__ half* src, half value, uint32_t count)
  __aicore__ inline void asc_max_scalar_sync(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t value, uint32_t count)
   __aicore__ inline void asc_max_scalar_sync(__ubuf__ float* dst, __ubuf__ float* src, float value, uint32_t count)
  ```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| dst       | 输出    | 目的操作数（矢量）的起始地址。            |
| src | 输入    | 源操作数（矢量）的起始地址。           |
| value | 输入    | 源操作数（标量）。       |
| count     | 输入    | 参与连续计算的元素个数。      |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr int total_length = 256; //total_length指参与计算的数据长度
half value = 3;
__ubuf__ half src[total_length];
__ubuf__ half dst[total_length];
asc_max_scalar_sync(dst, src, value, total_length);
// src : [1, 2, 3, 4, ... , 256]
// value : 3
// dst : [3, 3, 3, 5, ... , 256]
```
