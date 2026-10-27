---
type: API
title: asc_get_ar_spr (废弃)
description: 头文件路径为："c_api/spr/spr.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_get_ar_spr_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/spr/spr.h
---

# asc_get_ar_spr (废弃)

## 功能说明

头文件路径为：`"c_api/spr/spr.h"`。

**该接口已废弃，请使用[asc_get_squeeze_status](../spr/asc_get_squeeze_status.md)实现此功能。**

读取指定特殊寄存器的值。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ar_spr()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的特殊寄存器中的数值。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无
