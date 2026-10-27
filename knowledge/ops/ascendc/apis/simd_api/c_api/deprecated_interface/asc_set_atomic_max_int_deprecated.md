---
type: API
title: asc_set_atomic_max_int（废弃）
description: 该接口已废弃，请使用asc_set_atomic_max_int32替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_atomic_max_int_deprecated.md
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

# asc_set_atomic_max_int（废弃）

## 功能说明

头文件路径为：`"c_api/atomic/datamove_atomic.h"`。

**该接口已废弃，请使用[asc_set_atomic_max_int32](../atomic/datamove_atomic/asc_set_atomic_max.md)替代。**

设置计算结果以原子比较的方式传输到GM。在拷贝前，将待传输的int32_t数据与GM中已有数据进行逐元素比较，并将最大值写入GM。

## 函数原型

```c
[[deprecated("NOTICE: asc_set_atomic_max_int is deprecated. Please use asc_set_atomic_max_int32 instead.")]]
__aicore__ inline void asc_set_atomic_max_int()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

使用结束后，建议通过[asc_disable_dma_atomic](../atomic/datamove_atomic/asc_disable_dma_atomic.md)关闭原子最大操作，以免影响后续相关指令功能。

## 调用示例

请参考[asc_set_atomic_max](../atomic/datamove_atomic/asc_set_atomic_max.md)的调用示例。
