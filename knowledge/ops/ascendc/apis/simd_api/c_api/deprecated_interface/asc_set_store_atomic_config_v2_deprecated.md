---
type: API
title: asc_set_store_atomic_config_v2（废弃）
description: 该接口已废弃。原子加操作请使用asc_atomic_add实现。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_store_atomic_config_v2_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/atomic/scalar_atomic.h
---

# asc_set_store_atomic_config_v2（废弃）

## 功能说明

头文件路径为：`"c_api/atomic/scalar_atomic.h"`。

**该接口已废弃。原子加操作请使用[asc_atomic_add](../atomic/scalar_atomic/asc_atomic_add.md)实现。**

设置原子操作启用位与原子操作类型的值。

## 函数原型

```c
[[deprecated("NOTICE: asc_set_store_atomic_config_v2 is deprecated."
             "Please use asc_atomic_add instead for atomic add operation.")]]
__aicore__ inline void asc_set_store_atomic_config_v2(uint16_t type, uint16_t op)
```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| type | 输入 | 原子操作启用位。<br>1：开启原子操作，进行原子操作的数据类型为int32_t。<br>4：开启原子操作，进行原子操作的数据类型为half。<br>5：开启原子操作，进行原子操作的数据类型为float。<br>7：开启原子操作，进行原子操作的数据类型为bfloat16_t。<br>9：开启原子操作，进行原子操作的数据类型为int16_t。<br>10：开启原子操作，进行原子操作的数据类型为int8_t。<br>其余值无具体含义。 |
| op | 输入 | 原子操作类型。<br>2：求和操作。<br>其余值无具体含义。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c
uint16_t type = 1;
uint16_t op = 2;
asc_set_store_atomic_config_v2(type, op);
```
