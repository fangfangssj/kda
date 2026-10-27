---
type: API
title: make_coord
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/make_coord.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# make_coord

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

构造coord对象，用于定义张量的坐标。

## 函数原型

```cpp
template <typename FirstCoord, typename... Coords>
__aicore__ inline constexpr coord<FirstCoord, Coords...> make_coord(
    const FirstCoord& first_coord, const Coords&... coords)
```

## 参数说明

**表1**  模板参数类型说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| FirstCoord | 输入 | 第一个坐标的数据类型。支持size_t、int等整数类型，以及Std::Int类型。 |
| Coords... | 输入 | 其余坐标的数据类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| first_coord | 输入 | 第一个维度的坐标。 |
| coords... | 输入 | 其余维度的坐标。 |

## 返回值说明

返回`coord<FirstCoord, Coords...>`对象。

## 约束说明

- 各维度的坐标值需为非负整数。
- 参数数量必须与对应的shape维度数量一致。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 使用整数类型创建一个3维张量的坐标
auto coord = asc::te::make_coord(5, 10, 15);

// 获取各维度的坐标
auto coord0 = AscendC::Std::get<0>(coord); // coord0 = 5
auto coord1 = AscendC::Std::get<1>(coord); // coord1 = 10
auto coord2 = AscendC::Std::get<2>(coord); // coord2 = 15

// 使用Std::Int类型创建一个3维张量的坐标
auto coord_int = asc::te::make_coord(
    AscendC::Std::Int<5>{},
    AscendC::Std::Int<10>{},
    AscendC::Std::Int<15>{});

// 获取各维度的坐标
auto coord_int0 = AscendC::Std::get<0>(coord_int); // coord_int0 = 5
auto coord_int1 = AscendC::Std::get<1>(coord_int); // coord_int1 = 10
auto coord_int2 = AscendC::Std::get<2>(coord_int); // coord_int2 = 15
```
