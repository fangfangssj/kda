---
type: API
title: unalign_load
description: 头文件路径为："tensor_api/experimental/arch/vector/reg_data_load.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_data_load/unalign_load.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_data_load.h
- include/tensor_api/tensor.h
- include/tensor_api/experimental/vector_ops.h
---

# unalign_load

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_data_load.h"`，命名空间为`asc::te::experimental`。

`unalign_load`用于将非32字节对齐的Unified Buffer（UB）数据搬入[reg_tensor](../reg_tensor/reg_tensor.md)。对象内部维护一个32字节的非对齐搬入寄存器，提供以下两个操作：

- `prepare`：根据源Tensor起始地址初始化非对齐搬入寄存器。设源起始字节地址为`src_start_addr`，向低地址方向对齐到32字节边界得到`aligned_src_start_addr`，将`[aligned_src_start_addr, aligned_src_start_addr + 32)`的数据缓存到非对齐寄存器。该操作不会生成矢量数据寄存器。
- `load`：结合前置缓存，从指定UB地址搬入一个VL（Vector Length，矢量寄存器字节长度）的数据，支持从Tensor起始位置或指定坐标搬入。

## 函数原型

```cpp
struct unalign_load {
    __simd_callee__ unalign_load() = default;

    template <typename Tensor>
    __simd_callee__ inline void prepare(const Tensor& src_base);

    template <typename Tensor>
    __simd_callee__ inline auto load(const Tensor& src_tile);

    template <typename Tensor, typename Coord>
    __simd_callee__ inline auto load(const Tensor& src_tile, const Coord& coord);
};
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| Tensor | 源Tensor类型，由函数实参自动推导，必须位于UB。 |
| Coord | 源Tensor坐标类型，由函数实参自动推导，仅用于指定坐标的`load`重载。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src_base | 输入 | `prepare`使用的源Tensor，其起始地址必须与本次连续搬入序列中首次`load`的实际搬入地址相同。 |
| src_tile | 输入 | `load`使用的源Tensor。不传入`coord`时，从`src_tile.engine().begin().get()`搬入。 |
| coord | 输入 | 搬入起始元素在`src_tile`逻辑布局中的坐标，实际搬入地址为`(src_tile.data() + src_tile.layout()(coord)).get()`。 |

## 数据类型

| 源Tensor元素类型 | 返回的寄存器类型 |
| --- | --- |
| `int4b_t` | `reg_tensor<int4x2_t>`，以打包形式保存4bit数据。 |
| `fp4x2_e2m1_t`、`fp4x2_e1m2_t` | 对应元素类型的`reg_tensor`。 |
| `int8_t`、`uint8_t`、`hifloat8_t`、`fp8_e8m0_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t` | 对应元素类型的`reg_tensor`。 |
| `int16_t`、`uint16_t`、`half`、`bfloat16_t` | 对应元素类型的`reg_tensor`。 |
| `int32_t`、`uint32_t`、`float` | 对应元素类型的`reg_tensor`。 |
| `int64_t` | `reg_tensor<int64_t>`。 |

## 返回值说明

- `prepare`：无。
- `load`：返回包含一个VL数据的`reg_tensor`，仅初始化其`reg`成员，不初始化`mask`成员。后续使用依赖mask的计算或[store](../reg_data_store/store.md)前，必须通过`with_mask`等方式设置有效mask。

## 约束说明

- 起始地址无需32字节对齐，但必须满足元素类型的对齐要求。
- 同一连续搬入序列必须复用同一个`unalign_load`对象，并使用同一元素类型。序列开始前调用一次`prepare`，其源Tensor起始地址必须与首次`load`的实际搬入地址相同；后续搬入地址每次递增一个VL。地址不连续时，必须根据新的起始地址重新调用`prepare`。
- 指定`coord`时，坐标维数必须与Tensor布局匹配，坐标不能越界。
- `load`始终搬入一个VL数据，不根据Tensor形状截断，也不支持通过mask缩短实际读取范围。尾部不足一个VL时，仍须为完整读取及前置缓存访问预留合法UB空间；后续计算可使用mask屏蔽无效元素。
- `prepare`向低地址对齐后的32字节缓存访问范围以及`load`的实际读取范围，均必须位于有效UB地址空间内。不能仅根据逻辑Tensor大小判断访问安全性。

## 调用示例

### 从Tensor起始位置搬入

以下示例假设`src`和`dst`位于UB，起始地址按`uint8_t`对齐，源端已为完整VL读取和前置缓存预留合法空间，目的端至少有一个VL的可写空间。

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/vector_ops.h"

template <typename SrcTensor, typename DstTensor>
__simd_vf__ inline void CopyOneVector(SrcTensor src, DstTensor dst)
{
    asc::te::experimental::unalign_load loader;
    asc::te::experimental::unalign_store storer;
    loader.prepare(src);
    auto value = loader.load(src);
    constexpr uint32_t count = asc_get_vf_len() / sizeof(uint8_t);
    storer.store(dst, value, count);
    storer.flush(dst, static_cast<int32_t>(count));
}
```

### 从指定坐标搬入

`prepare`没有坐标参数，因此先复制Tensor视图，并将副本的引擎起始位置设置为本次坐标对应的地址。该操作不修改原Tensor，也不复制UB数据。源Tensor元素类型为`uint8_t`，实际UB空间必须足以覆盖从`offset`开始的完整VL读取。

```cpp
template <typename SrcTensor>
__simd_vf__ inline void LoadAtCoordinate(SrcTensor src, uint32_t offset)
{
    auto coord = asc::te::make_coord(offset);
    auto src_tile = src;
    src_tile.engine().begin() = src.data() + src.layout()(coord);
    asc::te::experimental::unalign_load loader;
    loader.prepare(src_tile);
    auto value = loader.load(src, coord);
    (void)value;
}
```
