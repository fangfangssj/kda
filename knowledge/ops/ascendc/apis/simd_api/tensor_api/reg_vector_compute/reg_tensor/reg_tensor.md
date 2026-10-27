---
type: API
title: reg_tensor
description: 头文件路径为："tensor_api/arch/vector/experimental/reg_tensor.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_tensor/reg_tensor.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_tensor.h
---

# reg_tensor

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_tensor.h"`，命名空间为`asc::te::experimental`。

`reg_tensor<DataType>`是矢量寄存器值的轻量级封装。同时保存矢量数据寄存器`reg`和掩码寄存器`mask`，当`DataType`为`bool`时，`reg_tensor<bool>`的reg和mask均为掩码寄存器。

## 定义原型

```cpp
template <typename DataType>
struct reg_tensor {
    using elem_type = DataType;
    using reg_type = typename detail::type_get<elem_type>::type;

    __simd_callee__ reg_tensor() = default;
    __simd_callee__ reg_tensor(const vector_bool& maskValue);

    __simd_callee__ inline reg_tensor& with_mask(const reg_tensor<bool>& maskReg);

    reg_type reg;
    vector_bool mask;
}
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| `DataType` | `reg_tensor`的元素类型，支持类型请参考[数据类型](#数据类型)。 |
| `maskValue` | 掩码寄存器值，带参构造函数将其写入`reg_tensor`对象的`mask`成员。 |

## 成员说明

| 成员名 | 描述 |
| --- | --- |
| `elem_type` | 元素类型，等同于`DataType`。 |
| `reg_type` | `DataType`对应的原生矢量寄存器类型。 |
| `reg` | 原生寄存器数据；对于`reg_tensor<bool>`，该成员保存谓词寄存器。 |
| `mask` | 与数据寄存器关联的谓词，供计算和单源`local_tensor::store`使用。 |

## 数据类型

| `DataType` | 对应的原生寄存器类型 |
| --- | --- |
| `bool` | `vector_bool` |
| `uint8_t`、`int8_t` | `vector_uint8_t`、`vector_int8_t` |
| `uint16_t`、`int16_t` | `vector_uint16_t`、`vector_int16_t` |
| `uint32_t`、`int32_t` | `vector_uint32_t`、`vector_int32_t` |
| `uint64_t`、`int64_t` | `vector_uint64_t`、`vector_int64_t` |
| `half`、`bfloat16_t`、`float` | `vector_half`、`vector_bfloat16_t`、`vector_float` |
| `hifloat8_t` | `vector_hifloat8_t` |
| `fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp8_e8m0_t` | 对应的`vector_fp8_*`类型 |
| `int4x2_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t` | 对应的`vector_int4x2_t`或`vector_fp4x2_*`类型 |

Tensor元素类型为`int4b_t`，寄存器对应的元素类型为`vector_int4x2_t`，`int4b_t`不作为`reg_tensor`的寄存器元素类型。

## 约束说明

- 默认构造函数不初始化`reg`和`mask`，使用前必须通过搬入、计算、赋值或mask绑定接口完成初始化。

## 成员函数

### with_mask

将输入`reg_tensor<bool>`寄存器的`maskReg.reg`绑定到当前`reg_tensor<DataType>`的`mask`成员，不修改数据成员`reg`。可与`load`接口链式调用。

#### 函数原型

```cpp
__simd_callee__ inline reg_tensor& with_mask(
    const reg_tensor<bool>& maskReg)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `maskReg` | 输入 | `reg_tensor<bool>`类型的寄存器，必须保证其`reg`成员已初始化。 |

#### 返回值说明

返回当前`reg_tensor`的引用。

#### 约束说明

`with_mask`只复制掩码寄存器，不校验掩码寄存器与数据元素宽度是否一致。

#### 调用示例

```cpp
auto value = asc::te::experimental::load(src, asc::te::make_coord(0))
                .with_mask(asc::te::experimental::all_mask<float>());
asc::te::experimental::store(dst, asc::te::make_coord(0), value);
```

## 调用示例

```cpp
__simd_vf__ inline void DeclareRegTensor()
{
    asc::te::experimental::reg_tensor<float> value;
    asc::te::experimental::reg_tensor<bool> predicate;
}
```
