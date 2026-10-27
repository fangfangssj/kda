---
type: API
title: axpy
description: 头文件路径："tensor_api/experimental/arch/vector/composite_compute.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/composite_compute/axpy.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/composite_compute.h
- include/tensor_api/experimental/vector_ops.h
---

# axpy

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/composite_compute.h"`。

该接口根据`dst.mask`将`src`与标量`scalar`按元素相乘，再与`dst`原值相加，返回计算结果。计算公式如下：

$$
dst_i = scalar \times src_i + dst_i
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> axpy(reg_tensor<T>& dst, const reg_tensor<T>& src, const T& scalar)
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输入/输出 | 加法操作数和结果，类型为`reg_tensor<T>`。接口直接更新`dst.reg`，`dst.mask`用于标记参与计算的元素。 |
| src | 输入 | 与标量相乘的源操作数，类型为`reg_tensor<T>`。`src.mask`不用于本次计算。 |
| scalar | 输入 | 标量乘数，类型必须与`T`完全一致。 |

## 返回值说明

返回`reg_tensor<T>`，返回值的`mask`与`dst.mask`相同。

## 数据类型

**表3** 数据类型组合

| src | scalar | dst |
| --- | --- | --- |
| half | half | half |
| float | float | float |

## 约束说明

- `dst.mask`需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。
- `dst.mask`对应位置为`0`时，返回值`reg`的对应位置置零。

## 调用示例

```cpp
#include "tensor_api/experimental/vector_ops.h"

template <typename DstTensor, typename SrcTensor>
__simd_vf__ inline void axpy_vf(
    DstTensor dst, const SrcTensor src, float scalar, uint16_t repeat_times, uint32_t total,
    uint32_t one_repeat_size)
{
    using data_type = typename DstTensor::data_type;
    uint32_t remain = total;
    for (uint16_t i = 0; i < repeat_times; ++i) {
        const uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto mask = asc::te::experimental::update_mask<data_type>(remain);
        auto dst_reg = asc::te::experimental::load(dst, coord).with_mask(mask);
        auto src_reg = asc::te::experimental::load(src, coord).with_mask(mask);
        dst_reg = asc::te::experimental::axpy(dst_reg, src_reg, scalar);
        asc::te::experimental::store(dst, coord, dst_reg);
    }
}
```
