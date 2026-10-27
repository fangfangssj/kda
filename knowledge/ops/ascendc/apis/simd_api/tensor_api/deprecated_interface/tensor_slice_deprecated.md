---
type: API
title: base_tensor::Slice（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/tensor_slice_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# base_tensor::Slice（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[base_tensor::slice](../tensor/tensor.md#slice)替代。**

从指定坐标处切出带有新Layout信息的子Tensor。

## 函数原型

```cpp
template <typename Coord, typename Info>
__aicore__ inline decltype(auto) Slice(const Coord& coord, const Info& info);

template <typename Coord, typename Info>
__aicore__ inline decltype(auto) Slice(const Coord& coord, const Info& info) const;
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| Coord | 输入 | 模板参数，表示切片起始坐标类型。 |
| Info | 输入 | 模板参数，表示切片描述类型，可以为Shape或Layout。 |
| coord | 输入 | 子Tensor在当前Tensor中的起始坐标。 |
| info | 输入 | 子Tensor的Shape或Layout描述。 |

## 返回值说明

返回切片后的子Tensor。新Tensor的Engine指向切片起始位置，Layout根据`coord`和`info`生成。

## 流水类型

无

## 约束说明

`coord`、`info`的结构及取值必须满足[slice](../tensor/slice.md)接口约束。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto old_slice = tensor.Slice(coord, info);  // 废弃写法
auto new_slice = tensor.slice(coord, info);  // 替代写法
```
