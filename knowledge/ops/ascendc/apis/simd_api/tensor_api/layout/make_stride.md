---
type: API
title: make_stride
description: 构造stride对象，用于描述各维度在内存中的步长。支持传入多个步长值，也支持传入嵌套的stride子结构以构造层次化stride。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/make_stride.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# make_stride

## 功能说明

构造stride对象，用于描述各维度在内存中的步长。支持传入多个步长值，也支持传入嵌套的stride子结构以构造层次化stride。

## 函数原型

```cpp
template <typename FirstStride, typename... Strides>
__aicore__ inline constexpr stride<FirstStride, Strides...> make_stride(
    const FirstStride& first_stride, const Strides&... strides)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| FirstStride | 输入 | 第一个stride元素的类型，可以是整型变量、`Std::Int`整型常量，或嵌套的stride子结构类型。 |
| Strides... | 输入 | 其余stride元素的类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| first_stride | 输入 | 第一个维度的步长或子结构。步长单位为元素个数。 |
| strides... | 输入 | 其余维度的步长或子结构。步长单位为元素个数。 |

## 返回值说明

返回`stride<FirstStride, Strides...>`对象。

## 约束说明

- 输入参数不可为空。
- 各参数必须为非负整数。
- 支持的数据类型包括：size_t、int等整数类型或者Std::Int类型。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 构造普通stride
auto stride = make_stride(1, 100, 200);

// 构造层次化stride
auto fractal_stride = make_stride(make_stride(1, 16), make_stride(32, 512));

auto stride0 = Std::get<0>(stride);                    // stride0 = 1
auto inner_stride = Std::get<1>(Std::get<0>(fractal_stride)); // inner_stride = 16
```
