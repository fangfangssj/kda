---
type: API
title: asc_transpose_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_transpose_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_transpose_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_transpose_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_permute_sel/asc_transpose.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

用于实现16*16的二维矩阵数据块转置。

## 函数原型

- 同步计算

  ```cpp
    __aicore__ inline void asc_transpose_sync(__ubuf__ int16_t* dst, __ubuf__ int16_t* src)
    __aicore__ inline void asc_transpose_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src)
  ```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|目的操作数（矢量）的起始地址。|
|src|输入|源操作数（矢量）的起始地址。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// total_length指参与计算的数据总长度
constexpr int total_length = 256;
__ubuf__ int16_t dst[total_length];
__ubuf__ int16_t src[total_length];
// dst指目的操作数的地址
asc_transpose_sync(dst, src);
```
