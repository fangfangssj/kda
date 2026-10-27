---
type: API
title: select
description: 头文件路径："include/tensor_api/experimental/arch/vector/compare_and_select.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/compare_and_select/select.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/compare_and_select.h
---

# select

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/compare_and_select.h"`。

给定两个源操作数src0和src1，根据condition的比特位值选取元素，得到目的操作数。当condition的比特位是1时，从src0中选取对应位置的数，比特位是0时从src1选取对应位置的数。

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline reg_tensor<T> select(
    const reg_tensor<bool>& condition, const reg_tensor<T>& src0, const reg_tensor<T>& src1)
```

## 参数说明

| 参数 | 输入/输出 | 描述 |
| --- | --- | --- |
| condition | 输入 | 选择条件，类型为 `reg_tensor<bool>`。每个元素决定对应位置选择 `src0` 还是 `src1`。 |
| src0 | 输入 | 条件为 `true` 时选择的源操作数，类型为 `reg_tensor<T>`。其 `mask` 作为返回值的有效元素掩码。 |
| src1 | 输入 | 条件为 `false` 时选择的源操作数，类型为 `reg_tensor<T>`。 |

## 返回值说明

返回 `reg_tensor<T>`，包含逐元素选择的结果。返回值的 `mask` 与 `src0.mask` 相同。

## 数据类型

支持的数据类型为：bool、int8_t、uint8_t、fp8_e4m3fn_t、hifloat8_t、fp8_e5m2_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。

## 约束说明

通过src0的mask参数控制的未选中的元素是未定义行为。

## 调用示例

```cpp
template <typename T, typename U, typename V>
__simd_vf__ inline void select_vf(T x_tensor, U y_tensor, V z_tensor,
    uint32_t repeat_times, uint32_t one_repeat_size)
{
    uint32_t count = TOTAL_LENGTH;
    using data_type = typename T::data_type;
    auto condition = asc::te::experimental::make_mask<asc::te::experimental::mask_pattern::vl8, data_type>();
    for (uint16_t i = 0; i < repeat_times; ++i) {
        auto mask = asc::te::experimental::update_mask<data_type>(count);
        uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto x_reg = asc::te::experimental::load(x_tensor, coord).with_mask(mask);
        auto y_reg = asc::te::experimental::load(y_tensor, coord);
        auto z_reg = asc::te::experimental::select(condition, x_reg, y_reg);
        asc::te::experimental::store(z_tensor, coord, z_reg);
    }
}
```
