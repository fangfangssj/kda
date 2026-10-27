---
type: API
title: make_shape
description: 构造shape对象，用于描述数据的逻辑形状。支持传入多个维度值，也支持传入嵌套的shape子结构以构造层次化shape。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/make_shape.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# make_shape

## 功能说明

构造shape对象，用于描述数据的逻辑形状。支持传入多个维度值，也支持传入嵌套的shape子结构以构造层次化shape。

## 函数原型

```cpp
template <typename FirstShape, typename... Shapes>
__aicore__ inline constexpr shape<FirstShape, Shapes...> make_shape(
    const FirstShape& first_shape, const Shapes&... shapes)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| FirstShape | 输入 | 第一个shape元素的类型，可以是整型变量、`Std::Int`整型常量，或嵌套的shape子结构类型。 |
| Shapes... | 输入 | 其余shape元素的类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| first_shape | 输入 | 第一个维度取值或子结构。 |
| shapes... | 输入 | 其余维度取值或子结构。 |

## 返回值说明

返回`shape<FirstShape, Shapes...>`对象。

## 约束说明

- 输入参数不可为空
- 输入的参数为正整数。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 构造普通shape
auto shape = make_shape(10, 20, 30);

// 构造层次化shape
auto fractal_shape = make_shape(make_shape(16, 8), make_shape(32, 4));

auto dim0 = Std::get<0>(shape);                // dim0 = 10
auto inner_row = Std::get<0>(Std::get<0>(fractal_shape)); // inner_row = 16
```
