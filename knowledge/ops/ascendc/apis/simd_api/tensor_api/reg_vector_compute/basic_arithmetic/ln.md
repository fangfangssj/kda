---
type: API
title: ln
description: 头文件路径："tensor_api/experimental/arch/vector/basic_arithmetic.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/basic_arithmetic/ln.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/basic_arithmetic.h
- include/tensor_api/experimental/vector_ops.h
---

# ln

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/basic_arithmetic.h"`。

该接口根据src.mask，对源操作数src中的有效元素逐个求自然对数（以e为底），并返回计算结果。计算公式如下：

$$
dst_i = \ln(src_i)
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> ln(const reg_tensor<T>& src)
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src | 输入 | 源操作数，类型为reg_tensor&lt;T&gt;。其中，src.reg保存矢量数据，src.mask用于控制各元素是否参与计算。src.mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 返回值说明

返回自然对数计算结果，类型为reg_tensor&lt;T&gt;。返回值的mask与src.mask相同；src.mask对应位置为0时，返回值的对应元素置零。

## 数据类型

**表3** 数据类型组合

| src | dst |
| --- | --- |
| half | half |
| float | float |

## 约束说明

- src.mask需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。

## 调用示例

```cpp
#include "tensor_api/experimental/vector_ops.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void ln_vf(
    const InputTensor input, OutputTensor output, uint16_t repeat_times, uint32_t total,
    uint32_t one_repeat_size)
{
    using data_type = typename InputTensor::data_type;
    uint32_t remain = total;
    for (uint16_t i = 0; i < repeat_times; ++i) {
        const uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto mask = asc::te::experimental::update_mask<data_type>(remain);
        auto src_reg = asc::te::experimental::load(input, coord).with_mask(mask);
        auto dst_reg = asc::te::experimental::ln(src_reg);
        asc::te::experimental::store(output, coord, dst_reg);
    }
}
```
