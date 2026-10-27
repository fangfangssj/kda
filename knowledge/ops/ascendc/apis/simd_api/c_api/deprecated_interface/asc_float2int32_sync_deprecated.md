---
type: API
title: asc_float2int32_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_float2int32_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_float2int32_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_float2int32_rn_sync`、`asc_float2int32_rna_sync`、`asc_float2int32_rd_sync`、`asc_float2int32_ru_sync`和`asc_float2int32_rz_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_convert/asc_float2int32.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将float类型转化为int32_t类型，并支持多种舍入模式：

* RINT舍入模式：四舍六入五成双舍入
* ROUND舍入模式：四舍五入舍入
* FLOOR舍入模式：向负无穷舍入
* CEIL舍入模式：向正无穷舍入
* TRUNC舍入模式：向零舍入

## 函数原型

* 前n个数据计算

- 同步计算

  ```cpp
  // RINT舍入模式
  __aicore__ inline void asc_float2int32_rn_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // ROUND舍入模式
  __aicore__ inline void asc_float2int32_rna_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // FLOOR舍入模式
  __aicore__ inline void asc_float2int32_rd_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // CEIL舍入模式
  __aicore__ inline void asc_float2int32_ru_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  // TRUNC舍入模式
  __aicore__ inline void asc_float2int32_rz_sync(__ubuf__ int32_t* dst, __ubuf__ float* src, uint32_t count)
  ```

## 参数说明

**表1** 参数说明

| 参数名    | 输入/输出 | 描述                |
| ------ | ----- | ----------------- |
| dst    | 输出    | 目的操作数（矢量）的起始地址。 |
| src    | 输入    | 源操作数（矢量）的起始地址。   |
| count  | 输入    | 参与连续计算的元素个数。      |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
__ubuf__ int32_t dst[256];
__ubuf__ float src[256];
asc_float2int32_rn_sync(dst, src, 256);
```
