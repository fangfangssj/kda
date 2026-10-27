---
type: API
title: 编译期整型别名
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/utils/compile_time_integer_aliases.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# 编译期整型别名

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

Tensor API提供一组[Std::Int](../../../utils_api/cpp_stdlib/type_traits/integral_constant.md)的类型别名，用于在Layout、Shape、Stride和Coord中表达编译期整数。

## 原型定义

```cpp
using _0 = Std::Int<0>;
using _1 = Std::Int<1>;
using _2 = Std::Int<2>;
using _3 = Std::Int<3>;
using _4 = Std::Int<4>;
using _5 = Std::Int<5>;
using _6 = Std::Int<6>;
using _7 = Std::Int<7>;
using _8 = Std::Int<8>;
using _9 = Std::Int<9>;
using _10 = Std::Int<10>;
using _16 = Std::Int<16>;
using _24 = Std::Int<24>;
using _32 = Std::Int<32>;
using _64 = Std::Int<64>;
using _128 = Std::Int<128>;
using _256 = Std::Int<256>;
using _512 = Std::Int<512>;
using _1024 = Std::Int<1024>;
using _2048 = Std::Int<2048>;
using _4096 = Std::Int<4096>;
```

## 类型说明

**表1**  类型说明

| 类型 | 描述 |
| :--- | :--- |
| 原型中列出的`_N`类型 | 对指定整数值的`Std::Int`类型别名。可用类型及其对应整数值以[原型定义](#原型定义)中的列表为准。 |

编译期整型别名是数据类型，使用时需要通过`_N{}`的形式构造对象。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 使用编译期整数构造shape和stride。
auto tensor_layout = make_layout(make_shape(_8{}, _16{}), make_stride(_16{}, _1{}));
```
