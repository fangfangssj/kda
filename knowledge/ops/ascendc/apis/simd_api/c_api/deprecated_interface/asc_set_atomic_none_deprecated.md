---
type: API
title: asc_set_atomic_none（废弃）
description: 该接口已废弃，请使用asc_disable_dma_atomic替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_atomic_none_deprecated.md
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

# asc_set_atomic_none（废弃）

## 功能说明

头文件路径为：`"c_api/atomic/datamove_atomic.h"`。

**该接口已废弃，请使用[asc_disable_dma_atomic](../atomic/datamove_atomic/asc_disable_dma_atomic.md)替代。**

与[asc_set_atomic_add](../atomic/datamove_atomic/asc_set_atomic_add.md)、[asc_set_atomic_max](../atomic/datamove_atomic/asc_set_atomic_max.md)、[asc_set_atomic_min](../atomic/datamove_atomic/asc_set_atomic_min.md)接口配合使用，用于在完成原子操作后关闭原子操作，避免影响后续搬运。

## 函数原型

```c
[[deprecated("NOTICE: asc_set_atomic_none is deprecated. Please use asc_disable_dma_atomic instead.")]]
__aicore__ inline void asc_set_atomic_none()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

请参见[asc_disable_dma_atomic的调用示例](../atomic/datamove_atomic/asc_disable_dma_atomic.md#调用示例)。
