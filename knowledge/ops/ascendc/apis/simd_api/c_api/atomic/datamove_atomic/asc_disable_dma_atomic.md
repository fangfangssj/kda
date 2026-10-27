---
type: API
title: asc_disable_dma_atomic
description: 头文件路径为："c_api/atomic/atomic.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/atomic/datamove_atomic/asc_disable_dma_atomic.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/atomic/datamove_atomic.h
---

# asc_disable_dma_atomic

## 功能说明

头文件路径为：`"c_api/atomic/datamove_atomic.h"`。

与[asc_set_atomic_add](asc_set_atomic_add.md)、[asc_set_atomic_max](asc_set_atomic_max.md)、[asc_set_atomic_min](asc_set_atomic_min.md)接口配合使用，用于在完成原子操作后关闭原子操作，避免影响后续搬运。

## 函数原型

```c
__aicore__ inline void asc_disable_dma_atomic()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

`PIPE_S`

## 约束说明

无

## 调用示例

asc_disable_dma_atomic与[asc_set_atomic_add](asc_set_atomic_add.md)等原子操作接口配合使用，调用示例请参见[asc_set_atomic_add的调用示例](asc_set_atomic_add.md#调用示例)。
