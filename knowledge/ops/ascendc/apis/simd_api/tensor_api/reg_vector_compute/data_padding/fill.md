---
type: API
title: fill
description: 头文件路径："include/tensor_api/experimental/arch/vector/compare_and_select.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/data_padding/fill.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/data_padding.h
---

# fill

## 功能说明

头文件路径：`"tensor_api/experimental/arch/vector/data_padding.h"`。

支持scalar和tensor两种模式：

- scalar模式：将value广播到寄存器，并保存在目的寄存器中（如果有mask，则保存在目的寄存器中被mask筛选的位置）。
- tensor模式：将src的最低位元素广播到寄存器，并保存在目的寄存器中被mask筛选的位置。

## 函数原型

- 将value广播到目的寄存器

    ```cpp
    template <size_t N = 1, typename T>
    __simd_callee__ inline reg_tensor<T> fill(const T& value, const reg_tensor<bool>& mask = all_mask<T>());
    ```

- 将寄存器src的最低位元素广播到目的寄存器

    ```cpp
    template <size_t N = 1, typename T>
    __simd_callee__ inline reg_tensor<T> fill(const reg_tensor<T>& src, const reg_tensor<bool>& mask = all_mask<T>());
    ```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| N | 待填充的寄存器数量，默认值为1。 |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |

**表2**  函数参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| value | 输入 | 源操作数。<br>类型为标量。 |
| src | 输入 | 源操作数。<br>类型为[reg_tensor]。 |
| mask | 输入 | 源操作数中元素有效性的指示，默认值为`all_mask<T>()`。 |

## 数据类型

支持的数据类型为：int8_t、uint8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。

## 返回值说明

返回 `reg_tensor<T>`，包含数据广播后的结果。返回值的 `mask` 与 输入的 `mask` 相同。

## 约束说明

- N当前仅支持1。
- mask仅筛选dst中写入广播值的位置，不筛选src中的元素。无论mask的最低位是否有效，待广播的数据均为src的最低位元素。

## 调用示例

- 示例一

    ```cpp
    template <typename T, typename U>
    __simd_vf__ inline void fill_vf(T y_tensor, U scalar, uint32_t repeat_times, uint32_t one_repeat_size)
    {
        uint32_t count = total_length;
        using data_type = typename T::data_type;
        for (uint16_t i = 0; i < repeat_times; ++i) {
            auto mask = asc::te::experimental::update_mask<data_type>(count);
            uint32_t offset = i * one_repeat_size;
            const auto coord = asc::te::make_coord(offset);
            auto y_reg = asc::te::experimental::fill(scalar, mask);
            asc::te::experimental::store(y_tensor, coord, y_reg);
        }
    }
    ```

- 示例二

    ```cpp
    template <typename T, typename U>
    __simd_vf__ inline void fill_vf(T x_tensor, U y_tensor, uint32_t repeat_times, uint32_t one_repeat_size)
    {
        uint32_t count = total_length;
        using data_type = typename T::data_type;
        for (uint16_t i = 0; i < repeat_times; ++i) {
            auto mask = asc::te::experimental::update_mask<data_type>(count);
            uint32_t offset = i * one_repeat_size;
            const auto coord = asc::te::make_coord(offset);
            auto x_reg = asc::te::experimental::load(x_tensor, coord);
            auto y_reg = asc::te::experimental::fill(x_reg, mask);
            asc::te::experimental::store(y_tensor, coord, y_reg);
        }
    }
    ```
