---
type: API
title: get_stride
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/get_stride.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# get_stride

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

get_stride用于从layout中提取stride部分：

- **不指定模板参数Is...**：返回完整的stride元组。
- **指定模板参数Is...**：从stride顶层tuple中选择对应索引的元素。指定单个索引时返回该元素；指定多个索引时返回由这些顶层元素组成的新tuple。

## 函数原型

```cpp
template <size_t... Is, typename Layout>
__aicore__ inline constexpr auto get_stride(const Layout& layout)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | ----------- | ------ |
| Is... | 输入 | 可选。用于选择stride顶层tuple中的一个或多个索引。 |
| Layout | 输入 | layout对象的类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | ----------- | ------ |
| layout | 输入 | layout对象。 |

## 返回值说明

返回描述内存访问步长的stride对象（或其子结构）。步长单位为元素个数，详见[stride](stride.md)。

## 约束说明

Is...必须在stride顶层tuple的有效范围内。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto layout = make_layout(make_shape(10, 20, 30), make_stride(1, 10, 200));

// 不指定Is... → 返回完整stride元组
auto full_stride = get_stride(layout);         // (1, 10, 200)

// 指定单个索引 → 返回对应元素
auto s0 = get_stride<0>(layout);             // 1

// 指定多个索引 → 组成新tuple
auto s01 = get_stride<0, 1>(layout);         // (1, 10)
```
