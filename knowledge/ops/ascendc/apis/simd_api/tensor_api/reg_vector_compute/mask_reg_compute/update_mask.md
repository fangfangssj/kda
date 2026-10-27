---
type: API
title: update_mask
description: 头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/mask_reg_compute/update_mask.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
---

# update_mask

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/mask_reg_compute.h"`，命名空间为`asc::te::experimental`。

根据当前剩余待处理元素数的值生成对应长度的有效位掩码，并自动将前剩余待处理元素数减去当前向量长度以更新剩余待处理元素数量。

## 函数原型

```cpp
template <typename DataType>
__simd_callee__ inline reg_tensor<bool> update_mask(uint32_t& remain)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `DataType` | 输入 | 模板参数，支持的数据类型为b8/b16/b32。 |
| `remain` | 输入/输出 | 元素个数。调用后自动减去当前向量处理单元的元素个数（b8模式、b16模式、b32模式分别为256、128、64）。<br>执行完一次该接口后，`remain = (remain < VL_T) ? 0 : (remain - VL_T)`，`VL_T`表示位宽为VL的矢量数据寄存器中可存放数据类型T的元素个数。 |

## 返回值说明

返回`reg_tensor<bool>`，其`reg`成员是类型为vector_bool的掩码寄存器，其对应的元素为有效数据。

## 约束说明

- 返回对象只保证`reg`成员有效，并不会初始化成员变量`mask`。
- 掩码寄存器数量上限为8。超出限制上限的寄存器数据会写入预留的8KUB内存中，可能会引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。

## 调用示例

```cpp
uint32_t remain = count;
for (uint32_t offset = 0; offset < count; offset += elementsPerReg) {
    auto tailMask = asc::te::experimental::update_mask<float>(remain);
    auto value = asc::te::experimental::load(src, asc::te::make_coord(offset)).with_mask(tailMask);
    asc::te::experimental::store(dst, asc::te::make_coord(offset), value);
}
```
