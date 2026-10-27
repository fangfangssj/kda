---
type: API
title: relu
description: 头文件路径："tensor_api/experimental/arch/vector/basic_arithmetic.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/basic_arithmetic/relu.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/basic_arithmetic.h
- include/tensor_api/tensor.h
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
- include/tensor_api/experimental/arch/vector/reg_data_load.h
- include/tensor_api/experimental/arch/vector/reg_data_store.h
---

# relu

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/basic_arithmetic.h"`。

该接口根据输入reg_tensor携带的mask，对源操作数按元素执行ReLU计算，并返回计算结果。计算公式如下：

$$
dst_i = \max(src_i, 0)
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> relu(const reg_tensor<T>& src)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src | 输入 | 源操作数，类型为reg_tensor&lt;T&gt;。其中，src.reg保存矢量数据，src.mask用于控制各元素是否参与计算。src.mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 数据类型

源操作数与返回值的数据类型保持一致。支持的数据类型为：half、int32_t、float。

## 返回值说明

返回按元素执行ReLU计算的结果，类型为reg_tensor&lt;T&gt;。返回值mask与src.mask相同；src.mask对应位置为0时，返回值的对应元素置零。

## 约束说明

src.mask需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。

## 调用示例

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/basic_arithmetic.h"
#include "tensor_api/experimental/arch/vector/mask_reg_compute.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void relu_example(InputTensor input, OutputTensor output)
{
    auto mask = asc::te::experimental::all_mask<float>();
    auto src = asc::te::experimental::load(input, asc::te::make_coord(0)).with_mask(mask);
    auto dst = asc::te::experimental::relu(src);
    asc::te::experimental::store(output, asc::te::make_coord(0), dst);
}
```
