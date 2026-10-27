---
type: API
title: asc_fma_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_fma_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_fma_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_fma_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_fused/asc_fma.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

按元素将src0和src1相乘并和dst相加，将最终结果存放进dst中，计算公式如下：
$$
dst_i = (src0_i * src1_i) + dst_i
$$

## 函数原型

- 同步计算

  ```cpp
  __aicore__ inline void asc_fma_sync(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
  __aicore__ inline void asc_fma_sync(__ubuf__ float* dst, __ubuf__ half* src0, __ubuf__ half* src1, uint32_t count)
  __aicore__ inline void asc_fma_sync(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t count)
  ```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| dst       | 输出    | 目的操作数（矢量）的起始地址。 |
| src0      | 输入    | 源操作数（矢量）的起始地址。 |
| src1      | 输入    | 源操作数（矢量）的起始地址。 |
| count     | 输入    | 参与计算的元素个数。        |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src0、src1的起始地址需要32字节对齐。

## 调用示例

```cpp
__ubuf__ half dst[256];
__ubuf__ half src0[256];
__ubuf__ half src1[256];
asc_fma_sync(dst, src0, src1, 256);
```
