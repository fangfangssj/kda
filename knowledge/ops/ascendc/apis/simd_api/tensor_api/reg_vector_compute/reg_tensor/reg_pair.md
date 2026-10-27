---
type: API
title: reg_pair
description: 头文件路径为："tensor_api/arch/vector/experimental/reg_tensor.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_tensor/reg_pair.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_tensor.h
---

# reg_pair

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_tensor.h"`，命名空间为`asc::te::experimental`。

保存两个元素类型可以相同或不同的`reg_tensor`，用于`deinterleave`等双结果接口。`first`的元素类型由`DataType`指定，`second`的元素类型由`CarryType`指定。`CarryType`默认与`DataType`相同。

## 定义原型

```cpp
template <typename DataType, typename CarryType = DataType>
struct reg_pair {
    reg_tensor<DataType> first;
    reg_tensor<CarryType> second;
};
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| `DataType` | 第一个寄存器Tensor的元素类型。 |
| `CarryType` | 第二个寄存器Tensor的元素类型，默认值为`DataType`。 |

## 成员说明

| 成员名 | 描述 |
| --- | --- |
| `first` | 第一个寄存器Tensor，类型为`reg_tensor<DataType>`。 |
| `second` | 第二个寄存器Tensor，类型为`reg_tensor<CarryType>`。 |

## 调用示例

接收`deinterleave`接口返回的两个寄存器Tensor：

```cpp
auto result = asc::te::experimental::deinterleave<float>(mask0, mask1);
asc::te::experimental::reg_tensor<bool> first = result.first;
asc::te::experimental::reg_tensor<bool> second = result.second;
```

接收[add_carry](../basic_arithmetic/add_carry.md)接口返回的两种元素类型的寄存器Tensor，`first`保存`uint32_t`类型的加法结果低32位，`second`保存`bool`类型的进位。

```cpp
auto src0 = asc::te::experimental::fill(uint32_t{0xffffffff});
auto src1 = asc::te::experimental::fill(uint32_t{1});
asc::te::experimental::reg_pair<uint32_t, bool> result =
    asc::te::experimental::add_carry(src0, src1);
asc::te::experimental::reg_tensor<uint32_t> sum = result.first;
asc::te::experimental::reg_tensor<bool> carry = result.second;
```
