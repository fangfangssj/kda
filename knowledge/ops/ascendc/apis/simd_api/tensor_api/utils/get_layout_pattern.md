---
type: API
title: get_layout_pattern
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/utils/get_layout_pattern.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# get_layout_pattern

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

get_layout_pattern用于从layout类型中提取其LayoutPattern类型信息。

## 原型定义

```cpp
template <typename Layout>
using get_layout_pattern = typename get_pattern<Std::remove_cvref_t<Layout>>::type;
```

## 参数说明

模板参数说明

**表1**  模板参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| Layout | 输入 | 待提取LayoutPattern的layout类型。 |

## 返回值说明

若Layout包含layout布局模式信息，则返回对应的LayoutPattern类型。

若Layout为make_layout构造的layout，get_layout_pattern会根据shape和stride的类型结构自动推导可识别的LayoutPattern；无法根据类型唯一识别时返回Std::ignore_t。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

using layout_type = decltype(make_frame_layout<nz_layout_ptn, layout_trait_default<float>>(16, 16));
using pattern = get_layout_pattern<layout_type>;
// pattern = nz_layout_ptn

using inferred_layout = decltype(make_layout(make_shape(asc::te::_16{}, asc::te::_16{})));
using inferred_pattern = get_layout_pattern<inferred_layout>;
// inferred_pattern = nd_layout_ptn

using plain_layout = decltype(make_layout(make_shape(16, 16), make_stride(16, 1)));
using plain_pattern = get_layout_pattern<plain_layout>;
// plain_pattern = Std::ignore_t，运行时数值未编码到类型中时无法自动推导
```
