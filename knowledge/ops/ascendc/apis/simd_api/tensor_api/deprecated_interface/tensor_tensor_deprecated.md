---
type: API
title: base_tensor::Tensor（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/tensor_tensor_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# base_tensor::Tensor（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[base_tensor::tensor](../tensor/tensor.md#tensor)替代。**

获取当前Tensor对象自身。

## 函数原型

```cpp
__aicore__ inline constexpr decltype(auto) Tensor() const;
```

## 参数说明

无

## 返回值说明

返回当前Tensor对象的常量引用。

## 流水类型

无

## 约束说明

无。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

const auto& old_tensor = tensor.Tensor();    // 废弃写法
const auto& new_tensor = tensor.tensor();    // 替代写法
```
