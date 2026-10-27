---
type: API
title: madd
description: 头文件路径："tensor_api/experimental/arch/vector/composite_compute.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/composite_compute/madd.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/composite_compute.h
- include/tensor_api/experimental/vector_ops.h
---

# madd

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/composite_compute.h"`。

该接口根据`dst.mask`将`dst`与`src0`按元素相乘，再与`src1`相加，返回计算结果。计算公式如下：

$$
dst_i = dst_i \times src0_i + src1_i
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> madd(reg_tensor<T>& dst, const reg_tensor<T>& src0, const reg_tensor<T>& src1)
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输入/输出 | 第一个乘数和结果。接口直接更新`dst.reg`，`dst.mask`控制参与计算的元素。 |
| src0 | 输入 | 第二个乘数。`src0.mask`不参与本次计算。 |
| src1 | 输入 | 加数。`src1.mask`不参与本次计算。 |

## 返回值说明

返回`reg_tensor<T>`，保存`dst * src0 + src1`的结果，返回值的`mask`与`dst.mask`相同。

## 数据类型

**表3** 数据类型组合

| src0 | src1 | dst |
| --- | --- | --- |
| half | half | half |
| bfloat16_t | bfloat16_t | bfloat16_t |
| float | float | float |

## 约束说明

- `dst.mask`需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。
- `dst.mask`对应位置为`0`时，返回值`reg`的对应位置置零。

## 调用示例

```cpp
#include "tensor_api/experimental/vector_ops.h"

template <typename DstTensor, typename Src0Tensor, typename Src1Tensor>
__simd_vf__ inline void madd_vf(
    DstTensor dst, const Src0Tensor src0, const Src1Tensor src1, uint16_t repeat_times, uint32_t total,
    uint32_t one_repeat_size)
{
    using data_type = typename DstTensor::data_type;
    uint32_t remain = total;
    for (uint16_t i = 0; i < repeat_times; ++i) {
        const uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto mask = asc::te::experimental::update_mask<data_type>(remain);
        auto dst_reg = asc::te::experimental::load(dst, coord).with_mask(mask);
        auto src0_reg = asc::te::experimental::load(src0, coord).with_mask(mask);
        auto src1_reg = asc::te::experimental::load(src1, coord).with_mask(mask);
        dst_reg = asc::te::experimental::madd(dst_reg, src0_reg, src1_reg);
        asc::te::experimental::store(dst, coord, dst_reg);
    }
}
```
