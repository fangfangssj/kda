---
type: API
title: operator*
description: 头文件路径："tensor_api/experimental/vector_compute.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/basic_arithmetic/operator_mul.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/basic_arithmetic.h
- include/tensor_api/experimental/vector_ops.h
---

# operator*

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/basic_arithmetic.h"`。

该接口根据输入reg_tensor携带的mask，对两个源操作数进行按元素相乘操作，并返回计算结果。支持reg_tensor与reg_tensor、reg_tensor与标量、标量与reg_tensor三种操作数组合。计算公式如下：

$$
dst_i = src0_i \times src1_i
$$

## 函数原型

- reg_tensor与reg_tensor

    ```cpp
    template <typename T>
    __simd_callee__ inline reg_tensor<T> operator*(
        const reg_tensor<T>& src0, const reg_tensor<T>& src1)
    ```

- reg_tensor与标量

    ```cpp
    template <typename T>
    __simd_callee__ inline reg_tensor<T> operator*(
        const reg_tensor<T>& src, const T& scalar)

    template <typename T>
    __simd_callee__ inline reg_tensor<T> operator*(
        const T& scalar, const reg_tensor<T>& src)
    ```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src0 | 输入 | reg_tensor与reg_tensor形式中的源操作数，类型为reg_tensor&lt;T&gt;。其中，src0.reg保存矢量数据，src0.mask用于控制各元素是否参与计算。src0.mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |
| src1 | 输入 | reg_tensor与reg_tensor形式中的源操作数，类型为reg_tensor&lt;T&gt;。src1.mask不参与本次计算。 |
| src | 输入 | reg_tensor与标量形式中的源操作数，类型为reg_tensor&lt;T&gt;。其中，src.reg保存矢量数据，src.mask用于控制各元素是否参与计算。src.mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |
| scalar | 输入 | 源操作数，类型为标量T。 |

## 数据类型

- reg_tensor与reg_tensor形式支持的数据类型为：int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。
- reg_tensor与标量形式支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。

## 返回值说明

返回逐元素相乘结果，类型为reg_tensor&lt;T&gt;。reg_tensor与reg_tensor形式的返回值mask与src0.mask相同，包含标量的形式与src.mask相同；输入mask对应位置为0时，返回值的对应元素置零。

## 约束说明

- 用于控制计算的mask需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。
- 输出结果为不饱和模式。

## 调用示例

```cpp
#include "tensor_api/experimental/vector_ops.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void mul_example(InputTensor input, OutputTensor output)
{
    using asc::te::experimental::operator*;
    auto src = asc::te::experimental::load(input, asc::te::make_coord(0))
                   .with_mask(asc::te::experimental::all_mask<float>());
    auto dst = src * 2.0f;
    asc::te::experimental::store(output, asc::te::make_coord(0), dst);
}
```
