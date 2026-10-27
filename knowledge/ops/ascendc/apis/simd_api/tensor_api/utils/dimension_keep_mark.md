---
type: API
title: 维度保留标记
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/utils/dimension_keep_mark.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# 维度保留标记

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

Tensor API提供`keep_mark`类型和常量`_`，当前仅用于在`squeeze`接口中标记需要保留的维度。

## 原型定义

```cpp
struct keep_mark {
    __aicore__ inline constexpr keep_mark() = default;
};

inline constexpr keep_mark _{};
```

## 类型说明

**表1**  类型说明

| 类型或常量 | 描述 |
| :--- | :--- |
| `keep_mark` | 维度保留标记类型。 |
| `_` | `keep_mark`类型的全局编译期常量，用于表示保留对应维度。 |

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 删除大小为1的第二个维度，保留第一个和第三个维度。
auto source_layout = make_layout(make_shape(4, _1{}, 8));
auto squeezed_layout = squeeze(source_layout, make_coord(_, _1{}, _));
```
