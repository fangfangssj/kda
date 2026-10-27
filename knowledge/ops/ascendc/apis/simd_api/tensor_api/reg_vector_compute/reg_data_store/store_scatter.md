---
type: API
title: store_scatter
description: 头文件路径为："tensor_api/experimental/arch/vector/reg_data_store.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_data_store/store_scatter.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_data_store.h
---

# store_scatter

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_data_store.h"`，命名空间为`asc::te::experimental`。

`store_scatter`根据索引寄存器，将[reg_tensor](../reg_tensor/reg_tensor.md)中的有效元素分散写入Unified Buffer（UB）Tensor。

传入`coord`时，以`tensor.engine() + tensor.layout()(coord)`为基地址；不传入`coord`时，以`tensor`起始地址为基地址。

## 函数原型

```cpp
template <typename Tensor, typename T, typename Index>
__simd_callee__ inline void store_scatter(
    Tensor& tensor,
    const reg_tensor<T>& src,
    const reg_tensor<Index>& index)

template <typename Tensor, typename Coord, typename T, typename Index>
__simd_callee__ inline void store_scatter(
    Tensor& tensor,
    const Coord& coord,
    const reg_tensor<T>& src,
    const reg_tensor<Index>& index)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `tensor` | 输出 | 位于UB的目的Tensor。 |
| `coord` | 输入 | 基地址在`tensor`逻辑布局中的坐标。 |
| `src` | 输入 | 源寄存器。`src.mask`逐元素指示是否写入。 |
| `index` | 输入 | `src`中各元素相对于基地址的目的位置，单位为元素。 |

## 数据类型

Tensor元素类型与`src`元素类型必须一致，支持的数据类型组合如下：

| Tensor元素类型/`src`元素类型 | `Index`类型 |
| --- | --- |
| `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t` | `uint16_t` |
| `int32_t`、`uint32_t`、`float` | `uint32_t` |

未列出的类型组合会产生编译期错误。

## 返回值说明

无。

## 约束说明

- 仅支持位于UB的Tensor，基地址必须32B对齐。
- Tensor元素类型必须与`src`元素类型一致。
- 索引对应的数据必须在UB有效地址范围内。
- `index`中的有效索引必须唯一。存在重复索引时，无法确定最终写入的数据。
- 当`src`元素类型为`int8_t`或`uint8_t`时，仅`src`中偶数位置的元素有效，即`src`中位置为0、2、4、...、252、254的元素会被分散搬出到目的操作数中。
- 传入`coord`时，坐标维数必须与Tensor布局匹配，且坐标不能越界。
- 寄存器数据搬出应在`__simd_vf__`函数或`__simd_callee__`函数中完成。

## 调用示例

```cpp
template <typename SrcTensor, typename IndexTensor, typename DstTensor>
__simd_vf__ inline void ScatterData(SrcTensor src, IndexTensor index,
    DstTensor dst, uint32_t repeat_times, uint32_t total_length,
    uint32_t one_repeat_size)
{
    using data_type = typename SrcTensor::data_type;
    uint32_t count = total_length;
    for (uint16_t i = 0; i < repeat_times; ++i) {
        auto mask = asc::te::experimental::update_mask<data_type>(count);
        uint32_t offset = i * one_repeat_size;
        const auto coord = asc::te::make_coord(offset);
        auto src_reg = asc::te::experimental::load(src, coord).with_mask(mask);
        auto index_reg = asc::te::experimental::load(index);
        asc::te::experimental::store_scatter(dst, coord, src_reg, index_reg);
    }
}
```
