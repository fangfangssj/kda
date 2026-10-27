---
type: API
title: none_mask
description: 头文件路径为："tensor_api/arch/vector/experimental/mask_reg_compute.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/mask_reg_compute/none_mask.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
---

# none_mask

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/mask_reg_compute.h"`，命名空间为`asc::te::experimental`。

创建`reg_tensor<bool>`，其成员变量reg是与`DataType`元素位宽匹配的全无效掩码寄存器，等价于`make_mask<mask_pattern::none, DataType>()`。

## 函数原型

```cpp
template <typename DataType>
__simd_callee__ inline reg_tensor<bool> none_mask()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `DataType` | 输入 | 模板参数，支持的数据类型为b8/b16/b32。 |

## 返回值说明

返回`reg_tensor<bool>`，其`reg`成员是类型为vector_bool的掩码寄存器，其对应的元素全为无效数据。

## 约束说明

- 返回对象只保证`reg`成员有效，并不会初始化成员变量`mask`。
- 掩码寄存器数量上限为8。超出限制上限的寄存器数据会写入预留的8KUB内存中，可能会引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。

## 调用示例

```cpp
auto disabled = asc::te::experimental::none_mask<uint32_t>();
auto value = asc::te::experimental::load(src, asc::te::make_coord(0)).with_mask(disabled);
```
