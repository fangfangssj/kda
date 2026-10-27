---
type: API
title: deinterleave(掩码解交织)
description: 头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/mask_reg_compute/deinterleave.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
---

# deinterleave(掩码解交织)

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/mask_reg_compute.h"`，命名空间为`asc::te::experimental`。

将两个源`reg_tensor<bool>`寄存器的reg按`DataType`对应的元素宽度解交织。`src0`和`src1`中偶数索引的分组依次写入返回值的`first.reg`，奇数索引的分组依次写入`second.reg`。

对于b8、b16和b32数据类型，接口分别以1 bit、2 bit和4 bit为一组进行解交织。`src0`产生返回结果的前半部分，`src1`产生返回结果的后半部分。

本接口只处理`reg_tensor<bool>`寄存器。数据寄存器的解交织接口参见[deinterleave（数据寄存器解交织）](../data_reorder/deinterleave.md)。

## 函数原型

```cpp
template <typename DataType>
__simd_callee__ inline reg_pair<bool> deinterleave(
    reg_tensor<bool> src0, reg_tensor<bool> src1)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `DataType` | 输入 | 模板参数，用于确定Mask的元素分组宽度，支持b8、b16和b32数据类型。 |
| `src0` | 输入 | 第一个源Mask寄存器，类型为`reg_tensor<bool>`，必须保证其`reg`成员已初始化。其偶数索引分组写入`first.reg`前半部分，奇数索引分组写入`second.reg`前半部分。 |
| `src1` | 输入 | 第二个源Mask寄存器，类型为`reg_tensor<bool>`，必须保证其`reg`成员已初始化。其偶数索引分组写入`first.reg`后半部分，奇数索引分组写入`second.reg`后半部分。 |

## 返回值说明

返回`reg_pair<bool>`：

- `first.reg`保存解交织后的偶数索引分组。
- `second.reg`保存解交织后的奇数索引分组。

返回对象只保证`first.reg`和`second.reg`有效，不会初始化两者的`mask`成员。

## 约束说明

- `src0`和`src1`可以是同一个`reg_tensor<bool>`对象。
- 本接口应在Vector Function（`__simd_vf__`标记的函数）内调用。
- Mask寄存器数量上限为8。超出限制上限的寄存器数据会写入预留的8K Unified Buffer（UB）内存中，可能引起性能劣化。

## 调用示例

```cpp
auto mask0 = asc::te::experimental::make_mask<
    asc::te::experimental::mask_pattern::every3, float>();
auto mask1 = asc::te::experimental::make_mask<
    asc::te::experimental::mask_pattern::every4, float>();

auto restored = asc::te::experimental::deinterleave<float>(
    mask0, mask1);
asc::te::experimental::reg_tensor<bool> restoredMask0 = restored.first;
asc::te::experimental::reg_tensor<bool> restoredMask1 = restored.second;
```
