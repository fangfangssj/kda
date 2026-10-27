---
type: API
title: coord
description: coord用于定义张量的坐标，用于访问张量中特定位置的元素。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/coord.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/tensor_api/tensor/layout_interface.h
---

# coord

coord用于定义张量的坐标，用于访问张量中特定位置的元素。

头文件路径为：`tensor_api/tensor.h`。

## 原型定义

```cpp
template <typename... Coords>
using coord = Std::tuple<Coords...>;
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| Coords... | 输入 | 可变参数模板，表示各维度的坐标。|

## API映射关系

coord通过[make_coord](make_coord.md)函数创建，可通过[crd2idx](crd2idx.md)函数进行多维坐标到线性坐标的转换，可作为参数传入到layout的[operator()](layout.md)成员函数，Tensor的[slice](../tensor/tensor.md)成员函数以及[slice](../tensor/slice.md)接口中进行张量的偏移以及切片操作。
