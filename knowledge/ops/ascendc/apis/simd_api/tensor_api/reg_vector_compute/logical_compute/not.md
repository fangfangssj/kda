---
type: API
title: not
description: 头文件路径："tensor_api/experimental/arch/vector/logical_compute.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/logical_compute/not.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/logical_compute.h
- include/tensor_api/tensor.h
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
- include/tensor_api/experimental/arch/vector/reg_data_load.h
- include/tensor_api/experimental/arch/vector/reg_data_store.h
---

# not

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/logical_compute.h"`。

命名空间：`asc::te::experimental`。

对输入寄存器中参与计算的元素逐位取反。`reg_tensor<bool>`对布尔寄存器数据取反，其他类型对元素的二进制编码取反。

$$
dst_i = \mathord{\sim}src_i
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> operator!(const reg_tensor<T>& src)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `T` | 模板参数 | 输入和返回值的元素类型，支持`bool`、`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`half`、`int32_t`、`uint32_t`、`float`。 |
| `src` | 输入 | 源操作数，类型为`reg_tensor<T>`。`src.mask`用于标记参与计算的元素，对应位置为`1`时参与计算，为`0`时不参与计算。 |

## 返回值说明

返回`reg_tensor<T>`。`reg`保存逐元素按位取反结果，`src.mask`对应位置为`0`时，`reg`的对应位置置零。返回值的`mask`与`src.mask`相同。

## 约束说明

- 本接口不是数值零值判断。对于非`bool`类型，结果是输入二进制编码逐位取反后的同类型数据。
- `src.mask`需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。
- 参与计算的元素个数由矢量长度（VL）决定：`T`不为`bool`时为`VL / sizeof(T)`，`T`为`bool`时为`VL`个bit。
- `src.mask`未选中的位置不参与计算，返回值对应位置置零。

## 调用示例

以下示例展示接口在AIV计算函数中的典型调用方式。

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/logical_compute.h"
#include "tensor_api/experimental/arch/vector/mask_reg_compute.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void not_example(InputTensor input, OutputTensor output)
{
    using asc::te::experimental::operator!;

    auto mask = asc::te::experimental::all_mask<int32_t>();
    auto src = asc::te::experimental::load(input, asc::te::make_coord(0)).with_mask(mask);
    auto result = !src;
    asc::te::experimental::store(output, asc::te::make_coord(0), result);
}
```
