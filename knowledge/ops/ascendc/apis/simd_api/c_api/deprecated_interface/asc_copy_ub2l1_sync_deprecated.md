---
type: API
title: asc_copy_ub2l1_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_copy_ub2l1_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_copy_ub2l1_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_copy_ub2l1_sync`接口已废弃，请使用[asc_copy_ub2l1](../vector_datamove/asc_copy_ub2l1.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将一段连续数据从Unified Buffer（UB）搬运到L1 Buffer，并在搬运完成后执行同步。

本接口仅在AIV上生效。

## 函数原型

```cpp
__aicore__ inline void asc_copy_ub2l1_sync(__cbuf__ void* dst, __ubuf__ void* src, uint32_t size)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的L1 Buffer的起始地址。 |
| src | 输入 | 源UB的起始地址。 |
| size | 输入 | 搬运数据大小，单位为字节。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

- 本接口在非AIV上调用直接返回。
- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 同步计算包含同步等待。

## 调用示例

```cpp
constexpr uint32_t size = 32;
__ubuf__ uint8_t src[size];
__cbuf__ uint8_t dst[size];
asc_copy_ub2l1_sync(dst, src, size);
```
