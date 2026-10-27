---
type: API
title: asc_set_atomic_add_int（废弃）
description: 该接口已废弃，请使用asc_set_atomic_add_int32替代。 头文件路径："c_api/atomic/atomic.h"。
  设置对后续的从Unified Buffer/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为int32_t。
tags:
- task.lookup
- topic.quality.compatibility.deprecation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_atomic_add_int_deprecated.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/atomic/datamove_atomic.h
---

# asc_set_atomic_add_int（废弃）

## 功能说明

头文件路径为：`"c_api/atomic/datamove_atomic.h"`。

**该接口已废弃，请使用[asc_set_atomic_add_int32](../atomic/datamove_atomic/asc_set_atomic_add.md)替代。**

设置对后续的从Unified Buffer（UB）/L0C Buffer/L1 Buffer到Global Memory的数据传输开启原子累加。累加的数据类型为int32_t。

## 函数原型

```c
[[deprecated("NOTICE: asc_set_atomic_add_int is deprecated. Please use asc_set_atomic_add_int32 instead.")]]
__aicore__ inline void asc_set_atomic_add_int()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 使用完成后，建议清空原子操作的状态（详见[asc_disable_dma_atomic](../atomic/datamove_atomic/asc_disable_dma_atomic.md)），以免影响后续相关指令功能。
- 该指令执行前不会对GM的数据做清零操作，开发者需在需要时手动添加清零操作。

## 调用示例

请参考[asc_set_atomic_add](../atomic/datamove_atomic/asc_set_atomic_add.md)的调用示例。
