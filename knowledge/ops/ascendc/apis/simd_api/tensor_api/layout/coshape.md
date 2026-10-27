---
type: API
title: coshape
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/coshape.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# coshape

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

coshape表示layout陪域（codomain）区间的标量大小，即逻辑坐标映射到一维索引后，所有可能的索引值构成的区间大小。其计算公式为：

```cpp
coshape = Σ( (shape[i] - 1) × stride[i] ) + 1
```

## 函数原型

```cpp
template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto coshape(const Layout& layout)
```

## 参数说明

**表1**  模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Is... | size_t... | 索引序列，指定子维度范围，仅对选定维度进行计算。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | layout用于描述张量的布局。 |

## 返回值说明

返回layout的陪域（codomain）的形状。

## 约束说明

Is...必须为有效范围内的索引。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 示例1：基础计算
auto layout = make_layout(make_shape(10, 20), make_stride(1, 100));
auto coshape_result = coshape(layout);
// (10-1)*1 + (20-1)*100 + 1 = 9 + 1900 + 1 = 1910

// 示例2：指定子维度
auto coshape0_result = coshape<0>(layout);
// (10-1)*1 + 1 = 9 + 1 = 10

auto coshape1_result = coshape<1>(layout);
// (20-1)*100 + 1 = 1900 + 1 = 1901
```
