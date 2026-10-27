---
type: API
title: Nop
description: 通过输入延迟个数实现延迟若干个cycle。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/Nop.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_utils_intf.h
---

# Nop

## 功能说明

头文件路径为：`"basic_api/kernel_operator_utils_intf.h"`。

用户输入延迟的个数，实现延迟若干个cycle。

## 函数原型

```cpp
template <int count = 1>
__aicore__ inline void Nop()
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| count | 输入 | 延迟count个cycle。 |

## 约束说明

无

## 调用示例

如下样例中，延迟10个cycle。

```cpp
Nop<10>();
```
