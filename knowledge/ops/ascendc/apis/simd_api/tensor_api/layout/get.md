---
type: API
title: get
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/get.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# get

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

get提供两类能力：

1. 对元组Tuple，按位置索引提取元素或逐层提取嵌套元素；
2. 对layout，分别从shape和stride中提取对应位置的子tuple，再组合为新的子layout。

## 函数原型

```cpp
template <size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto get(Tuple&& t)

template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto get(const Layout& layout)
```

## 参数说明

**表1**  模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Tuple | 输入 | 元组，shape和stride都是Tuple的别名。 |
| I | size_t | 元组重载中第一个索引。 |
| Is... | size_t | 其余索引序列，用于编译时递归选择元组或layout的子结构。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| t | 输入 | 待提取元素的元组对象。 |
| layout | 输入 | layout用于描述张量的布局。 |

## 返回值说明

- 对输入为Tuple类型，返回按索引链逐层提取后的元素；
- 对输入为layout类型，返回由提取后的shape和stride构成的新子layout对象。

## 约束说明

Is...必须为有效范围内的索引。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto t = Std::make_tuple(1, Std::make_tuple(2, 3));

auto v0 = get<0>(t);
// v0 = 1

auto v1 = get<1, 0>(t);
// v1 = 2

auto layout = make_layout(make_shape(make_shape(10, 20), make_shape(20, 30)), make_stride(make_stride(10, 200), make_stride(1, 4000)));

// 提取第 1 维度
auto sub_layout = get<1>(layout);
// sub_layout  = make_layout(make_shape(20, 30), make_stride(1, 4000));
```
