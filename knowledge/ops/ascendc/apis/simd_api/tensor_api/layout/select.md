---
type: API
title: select
description: 选择layout的shape和stride指定维度组成新的layout对象并返回。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/select.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# select

## 功能说明

选择layout的shape和stride指定维度组成新的layout对象并返回。

## 函数原型

```cpp
template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto select(const Layout& layout)
```

## 参数说明

**表1**  模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Is... | size_t | 索引序列，用于编译时递归选择shape和stride的子结构。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | layout用于描述张量的布局。 |

## 返回值说明

返回子layout对象。

## 约束说明

索引Is...必须在有效范围内。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto shape = make_shape(10, 20, 30);
auto layout = make_layout(shape);

// 选择第0和第1维度
auto sub_layout = select<0, 1>(layout);
```
