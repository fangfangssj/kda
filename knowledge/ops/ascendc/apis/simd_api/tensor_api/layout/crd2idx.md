---
type: API
title: crd2idx
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/crd2idx.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# crd2idx

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

crd2idx根据coord、shape和stride计算坐标对应的一维线性索引。输入为嵌套Tuple时，接口会按照相同的层次结构递归计算各维度的偏移并求和。

## 函数原型

```cpp
template <typename Coord, typename Shape, typename Stride>
__aicore__ inline constexpr auto crd2idx(
    const Coord& coord, const Shape& shape, const Stride& stride)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| Coord | 输入 | 坐标类型，可以是整数、编译期整型常量或嵌套Tuple。 |
| Shape | 输入 | shape类型，可以是整数、编译期整型常量或嵌套Tuple。 |
| Stride | 输入 | stride类型，其结构需要与shape匹配。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| coord | 输入 | 待转换的坐标。 |
| shape | 输入 | 描述各维度大小的shape。 |
| stride | 输入 | 描述各维度步长的stride。 |

## 返回值说明

返回coord对应的一维线性索引，返回类型由coord和stride的计算结果推导。

## 约束说明

- coord、shape和stride为Tuple时，顶层元素数量需要一致，嵌套结构需要能够逐层对应。
- coord中的各维坐标需要位于shape描述的有效范围内。
- stride的单位与目标地址计算所使用的单位保持一致。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto tensor_shape = make_shape(8, 16);
auto tensor_stride = make_stride(16, 1);
auto tensor_coord = make_coord(2, 3);

auto index = crd2idx(tensor_coord, tensor_shape, tensor_stride);
// index = 2 * 16 + 3 = 35
```
