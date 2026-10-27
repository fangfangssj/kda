---
type: API
title: interleave（掩码交织）
description: 头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/mask_reg_compute/interleave.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
---

# interleave（掩码交织）
 
## 功能说明
 
头文件路径为：`"tensor_api/experimental/arch/vector/mask_reg_compute.h"`，命名空间为`asc::te::experimental`。
 
将两个`reg_tensor<bool>`寄存器的reg按`DataType`对应的元素宽度交织，并将交织结果的低半部分和高半部分分别保存在返回值的`first`和`second`中。
 
对于b8、b16和b32数据类型，接口分别以1 bit、2 bit和4 bit为一组进行交织。`src0`和`src1`前半部分的分组依次交替写入`first.reg`，后半部分的分组依次交替写入`second.reg`。
 
本接口只处理输入为`reg_tensor<bool>`类型的寄存器。数据寄存器的交织接口参见[数据重排](../data_reorder/index.md)。
 
## 函数原型
 
```cpp
template <typename DataType>
__simd_callee__ inline reg_pair<bool> interleave(
    reg_tensor<bool> src0, reg_tensor<bool> src1)
```
 
## 参数说明
 
| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| DataType | 输入 | 模板参数，用于确定Mask的元素分组宽度，支持b8、b16和b32数据类型。 |
| src0 | 输入 | 第一个源Mask寄存器，类型为`reg_tensor<bool>`，必须保证其`reg`成员已初始化。 |
| src1 | 输入 | 第二个源Mask寄存器，类型为`reg_tensor<bool>`，必须保证其`reg`成员已初始化。 |
 
## 返回值说明
 
返回`reg_pair<bool>`：
 
- `first.reg`保存交织结果的低半部分。
- `second.reg`保存交织结果的高半部分。
 
返回对象只保证`first.reg`和`second.reg`有效，不会初始化两者的`mask`成员。
 
## 约束说明
 
- `src0`和`src1`可以是同一个`reg_tensor<bool>`对象。
- 本接口应在Vector Function（`__simd_vf__`标记的函数）内调用。
- Mask寄存器数量上限为8。超出限制上限的寄存器数据会写入预留的8K UB内存中，可能引起性能劣化。
 
## 调用示例
 
```cpp
auto mask0 = asc::te::experimental::make_mask<
    asc::te::experimental::mask_pattern::every3, float>();
auto mask1 = asc::te::experimental::make_mask<
    asc::te::experimental::mask_pattern::every4, float>();
 
auto result = asc::te::experimental::interleave<float>(mask0, mask1);
asc::te::experimental::reg_tensor<bool> lowMask = result.first;
asc::te::experimental::reg_tensor<bool> highMask = result.second;
```
