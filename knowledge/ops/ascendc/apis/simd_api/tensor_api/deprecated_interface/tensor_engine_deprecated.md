---
type: API
title: base_tensor::Engine（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/tensor_engine_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# base_tensor::Engine（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[base_tensor::engine](../tensor/tensor.md#engine)替代。**

获取Tensor绑定的Engine对象。

## 函数原型

```cpp
__aicore__ inline constexpr decltype(auto) Engine() const;
__aicore__ inline constexpr decltype(auto) Engine();
```

## 参数说明

无

## 返回值说明

返回Engine对象的引用；const重载返回常量引用。

## 流水类型

无

## 约束说明

无。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto& old_engine = tensor.Engine();          // 废弃写法
auto& new_engine = tensor.engine();          // 替代写法
```
