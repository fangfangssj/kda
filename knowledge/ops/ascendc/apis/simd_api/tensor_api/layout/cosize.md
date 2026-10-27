---
type: API
title: cosize
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/cosize.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# cosize

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

cosize表示layout陪域（codomain）的跨度大小，即逻辑坐标经layout映射为一维索引后，所有可能索引值所处区间的长度。cosize和coshape在扁平与嵌套layout场景下等价。

## 函数原型

```cpp
template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto cosize(const Layout& layout)
```

## 参数说明

**表1**  模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Is... | size_t | 索引序列，可通过模板参数Is...指定子维度范围。用于编译时递归选择shape和stride的子结构。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | layout用于描述张量的布局。 |

## 返回值说明

返回陪域空间的元素个数。

## 约束说明

Is...必须为有效范围内的索引。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 示例1：扁平layout，cosize = coshape
auto layout = make_layout(make_shape(10, 20), make_stride(1, 100));
auto cosize_result = cosize(layout);
// coshape = (10-1)*1 + (20-1)*100 + 1 = 1910
// cosize = tuple_size(1910) = 1910

// 示例2：指定子维度
auto cosize0_result = cosize<0>(layout);
// coshape<0> = (10-1)*1 + 1 = 10
// cosize<0> = tuple_size(10) = 10
```
