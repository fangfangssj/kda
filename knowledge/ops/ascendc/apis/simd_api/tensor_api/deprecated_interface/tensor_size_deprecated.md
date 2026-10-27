---
type: API
title: base_tensor::Size（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/tensor_size_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# base_tensor::Size（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[base_tensor::size](../tensor/tensor.md#size)替代。**

获取Tensor覆盖的逻辑元素总数。

## 函数原型

```cpp
__aicore__ inline constexpr auto Size() const;
```

## 参数说明

无

## 返回值说明

返回`layout().size()`的计算结果。

## 流水类型

无

## 约束说明

无。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto old_size = tensor.Size();               // 废弃写法
auto new_size = tensor.size();               // 替代写法
```
