---
type: API
title: make_pattern_layout
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/layout/make_pattern_layout.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# make_pattern_layout

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

make_pattern_layout用于根据shape、stride、LayoutPattern和Trait构造带布局模式信息的layout对象。

与[make_layout](make_layout.md)相比，make_pattern_layout除了保存shape和stride外，还会在layout类型中附加布局模式LayoutPattern和Trait信息。

Trait用于描述C0粒度等布局特征。若使用`layout_trait<T, C0>`或`layout_trait_default<T>`写法，T仅作为兼容占位或C0推导依据，不作为Tensor的数据类型信息保存。

## 函数原型

```cpp
template <typename LayoutPattern, typename Trait, typename Shape, typename Stride>
__aicore__ inline constexpr auto make_pattern_layout(const Shape& shape, const Stride& stride)
```

## 参数说明

**表1**  模板参数说明

| 参数名称 | 类型 | 描述 |
| :------- | :--- | :--- |
| LayoutPattern | 输入 | 布局模式类型，用于标记layout的格式信息。 |
| Trait | 输入 | 布局trait类型，用于标记C0粒度等布局特征。 |
| Shape | 输入 | shape的类型，要求为元组(tuple)类型。 |
| Stride | 输入 | stride的类型，要求为元组(tuple)类型。 |

**表2**  参数说明

| 参数名称 | 输入/输出 | 描述 |
| :------- | :-------- | :--- |
| shape | 输入 | 描述逻辑形状的元组对象，可由[make_shape](make_shape.md)构造。单位为元素个数。 |
| stride | 输入 | 描述各维步长的元组对象，可由[make_stride](make_stride.md)构造。单位为元素个数。 |

## 返回值说明

返回`layout<Shape, Stride, Std::tuple<LayoutPattern, Trait>>`类型对象。

## 约束说明

- shape和stride必须满足layout构造要求，即都为元组类型。
- LayoutPattern应为合法的布局模式类型，例如nz_layout_ptn、zn_layout_ptn、nd_layout_ptn、dn_layout_ptn等。
- Trait应与LayoutPattern及后续使用场景匹配，否则可能导致后续格式检查或路由失败。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

// 示例1：构造带NZ布局模式的layout
auto layout1 = make_pattern_layout<nz_layout_ptn, layout_trait_default<>>(
    make_shape(make_shape(16, 2), make_shape(16, 4)),
    make_stride(make_stride(16, 256), make_stride(1, 512))
);

// 示例2：构造带ND布局模式的layout
auto layout2 = make_pattern_layout<nd_layout_ptn, layout_trait<Std::ignore_t, _1>>(
    make_shape(32, 64),
    make_stride(64, 1)
);
```
