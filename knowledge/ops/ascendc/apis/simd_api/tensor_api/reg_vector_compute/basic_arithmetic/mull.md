---
type: API
title: mull
description: 头文件路径："tensor_api/experimental/arch/vector/basic_arithmetic.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/basic_arithmetic/mull.md
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

# mull

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/basic_arithmetic.h"`。

该接口根据src0携带的mask，对两个32位整数源操作数按元素相乘，返回由低32位和高32位组成的结果。计算公式如下：

$$
\{high_i, low_i\} = src0_i \times src1_i
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_pair<T> mull(
    const reg_tensor<T>& src0, const reg_tensor<T>& src1)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src0 | 输入 | 源操作数，类型为reg_tensor&lt;T&gt;。src0.reg保存矢量数据，src0.mask控制各元素是否参与计算。 |
| src1 | 输入 | 源操作数，类型为reg_tensor&lt;T&gt;。src1.mask不参与本次计算。 |

## 数据类型

源操作数与返回值中两个寄存器的数据类型保持一致。支持的数据类型为：int32_t、uint32_t。

## 返回值说明

返回`reg_pair<T>`。`first`保存乘积的低32位，`second`保存乘积的高32位，两个寄存器的mask均与`src0.mask`相同。

## 约束说明

src0.mask需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。

## 调用示例

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/basic_arithmetic.h"
#include "tensor_api/experimental/arch/vector/mask_reg_compute.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void mull_example(InputTensor input0, InputTensor input1, OutputTensor output)
{
    auto mask = asc::te::experimental::all_mask<uint32_t>();
    auto src0 = asc::te::experimental::load(input0, asc::te::make_coord(0)).with_mask(mask);
    auto src1 = asc::te::experimental::load(input1, asc::te::make_coord(0));
    auto result = asc::te::experimental::mull(src0, src1);
    asc::te::experimental::store(output, asc::te::make_coord(0), result.first);
    asc::te::experimental::store(output, asc::te::make_coord(1), result.second);
}
```
