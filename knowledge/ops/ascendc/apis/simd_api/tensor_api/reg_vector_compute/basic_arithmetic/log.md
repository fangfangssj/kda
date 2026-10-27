---
type: API
title: log
description: 头文件路径："tensor_api/experimental/vector_compute.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/basic_arithmetic/log.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
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

# log

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/basic_arithmetic.h"`。

该接口根据`src.mask`，对源操作数`src`中的有效元素逐个求自然对数（以 e 为底），并返回计算结果。`log` 与 `ln` 的计算语义一致，计算公式如下：

$$
dst_i = \ln(src_i)
$$

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> log(const reg_tensor<T>& src)
```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src | 输入 | 源操作数，类型为reg_tensor&lt;T&gt;。其中，src.reg保存矢量数据，src.mask用于控制各元素是否参与计算。src.mask中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |

## 数据类型

源操作数与返回值的数据类型保持一致。支持的数据类型为：half、float。

## 返回值说明

返回自然对数计算结果，类型为reg_tensor&lt;T&gt;。返回值的mask与src.mask相同；src.mask对应位置为0时，返回值的对应元素置零。

## 约束说明

- `src.mask`需通过`with_mask`接口预先设置。未设置时，mask的内容不确定，会导致参与计算的元素位置错误。

## 调用示例

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/basic_arithmetic.h"
#include "tensor_api/experimental/arch/vector/mask_reg_compute.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

template <typename InputTensor, typename OutputTensor>
__simd_vf__ inline void log_example(InputTensor input, OutputTensor output)
{
    auto src = asc::te::experimental::load(input, asc::te::make_coord(0))
                       .with_mask(asc::te::experimental::all_mask<float>());
    auto dst = asc::te::experimental::log(src);
    asc::te::experimental::store(output, asc::te::make_coord(0), dst);
}
```
