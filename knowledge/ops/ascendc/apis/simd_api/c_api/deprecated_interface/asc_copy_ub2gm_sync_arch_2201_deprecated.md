---
type: API
title: asc_copy_ub2gm_sync_arch_2201（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_copy_ub2gm_sync_arch_2201_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_copy_ub2gm_sync_arch_2201（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_copy_ub2gm_sync`接口已废弃，请使用[对应的非同步接口](../vector_datamove/asc_copy_ub2gm/asc_copy_ub2gm_arch_2201.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将数据从Unified Buffer（UB）搬运到Global Memory (GM)。

## 函数原型

- 同步计算

```cpp
__aicore__ inline void asc_copy_ub2gm_sync(__gm__ void* dst, __ubuf__ void* src, uint32_t size)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的GM地址。 |
| src | 输入 | 源UB地址。 |
| size | 输入 | 搬运数据大小（字节）。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 如果本指令与其他指令存在UB地址重叠，必须插入同步指令[asc_sync_notify](../sync/intra_core_sync/asc_sync_notify.md)和[asc_sync_wait](../sync/intra_core_sync/asc_sync_wait.md)，保证多个指令串行化，防止出现异常数据。
- 同步计算包含同步等待。
- 当采用前n个数据搬运接口时，搬运数据大小要求32字节对齐。

## 调用示例

```cpp
// total_length指参与搬运的数据总长度。dst是外部输入的half类型的GM内存。
constexpr uint64_t total_length = 128;
__ubuf__ half src[total_length];
asc_copy_ub2gm_sync(dst, src, total_length * sizeof(half));
```
