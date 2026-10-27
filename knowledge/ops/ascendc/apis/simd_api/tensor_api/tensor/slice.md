---
type: API
title: slice
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/tensor/slice.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# slice

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

slice用于从张量指定坐标处切出一个子张量。根据输入的起始坐标coord和切片描述信息info，计算新的起始存储位置，并构造对应的子张量，子张量保持与原张量相同的存储位置类型。

## 函数原型

```cpp
template <typename Tensor, typename Coord, typename Info>
__aicore__ inline decltype(auto) slice(Tensor&& tensor, const Coord& coord, const Info& info)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| Tensor | 输入 | 张量类型，通常为global_tensor<...>或local_tensor<...>。 |
| Coord | 输入 | 坐标类型，通常为coord<...>元组类型。 |
| Info | 输入 | 切片描述类型，可以为shape类型或layout类型。传入shape时，表示按给定形状截取子张量；传入layout时，表示按layout中的shape信息截取子张量并保持原有布局模式和stride结构。 |

**表2**  参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| tensor | 输入 | 待切片的张量对象。 |
| coord | 输入 | 切片起始坐标。 |
| info | 输入 | 切片描述信息。可以传入shape或layout。 |

## 返回值说明

返回从coord指定位置开始的子张量对象。返回结果包含新的layout信息，Engine指向切片后的起始位置。

## 约束说明

- coord的tuple结构必须与当前Tensor的layout.shape()结构一致。
- 当info为shape时：info必须是二维shape；当前Tensor的layout必须为二维或四维。
- 当info为layout时，info必须与当前Tensor的layout的维度相同。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

constexpr uint64_t gm_addr = 128;

// 示例1：按shape切片
auto layout1 = make_layout(make_shape(10, 20), make_stride(20, 1));
auto tensor1 = make_tensor(make_mem_ptr<location::gm, float>(gm_addr), layout1);
auto sub1 = slice(tensor1, make_coord(2, 4), make_shape(4, 8));
// 起始坐标为(2, 4)，子张量形状为(4, 8)

// 示例2：按layout切片
auto layout2 = make_layout(make_shape(10, 20), make_stride(20, 1));
auto tensor2 = make_tensor(make_mem_ptr<location::gm, float>(gm_addr), layout2);
auto sub2 = slice(tensor2, make_coord(1, 2), make_layout(make_shape(3, 5), make_stride(20, 1)));
// 从(1, 2)开始切出一个shape为(3, 5)的子张量，并保持stride结构

// 示例3：等价于调用成员函数
auto sub3 = tensor2.slice(make_coord(1, 2), make_shape(3, 5));
// sub3与slice(tensor2, make_coord(1, 2), make_shape(3, 5))等价
```
