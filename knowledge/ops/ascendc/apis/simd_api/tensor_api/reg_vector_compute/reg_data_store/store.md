---
type: API
title: store
description: 头文件路径为："tensor_api/experimental/arch/vector/reg_data_store.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_data_store/store.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_data_store.h
- include/tensor_api/tensor.h
- include/tensor_api/experimental/arch/vector/mask_reg_compute.h
- include/tensor_api/experimental/arch/vector/reg_data_load.h
---

# store

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_data_store.h"`，命名空间为`asc::te::experimental`。

`store`用于将[reg_tensor](../reg_tensor/reg_tensor.md)中的数据搬出到Unified Buffer（UB）Tensor。接口通过`store_sideband_mode`配置正常搬出、首元素搬出、交织搬出或压缩搬出等数据排列方式。

接口支持以下两种寻址方式：

- 传入`coord`时，从`tensor.engine() + tensor.layout()(coord)`对应的元素开始写入。
- 不传入`coord`时，从`tensor`起始位置开始写入。

`intlv`模式使用双源操作数重载，其他模式使用单源操作数重载。

## 函数原型

- 单源操作数，指定坐标：

    ```cpp
    template <store_sideband_mode sideband_mode = store_sideband_mode::direct,
              typename Tensor,
              typename Coord,
              typename RegDataType>
    __simd_callee__ inline void store(
        Tensor& tensor,
        const Coord& coord,
        const reg_tensor<RegDataType>& src)
    ```

- 单源操作数，从Tensor起始位置搬出：

    ```cpp
    template <store_sideband_mode sideband_mode = store_sideband_mode::direct,
              typename Tensor,
              typename RegDataType>
    __simd_callee__ inline void store(
        Tensor& tensor, const reg_tensor<RegDataType>& src)
    ```

- 双源操作数，指定坐标：

    ```cpp
    template <store_sideband_mode sideband_mode = store_sideband_mode::intlv,
              typename Tensor,
              typename Coord,
              typename RegDataType>
    __simd_callee__ inline void store(
        Tensor& tensor,
        const Coord& coord,
        const reg_tensor<RegDataType>& src0,
        const reg_tensor<RegDataType>& src1)
    ```

- 双源操作数，从Tensor起始位置搬出：

    ```cpp
    template <store_sideband_mode sideband_mode = store_sideband_mode::intlv,
              typename Tensor,
              typename RegDataType>
    __simd_callee__ inline void store(
        Tensor& tensor,
        const reg_tensor<RegDataType>& src0,
        const reg_tensor<RegDataType>& src1)
    ```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sideband_mode | 搬出模式，类型为`store_sideband_mode`。单源操作数重载的默认值为`store_sideband_mode::direct`；双源操作数重载的默认值为`store_sideband_mode::intlv`。具体取值请参考[表3](#表3-store_sideband_mode参数说明)。 |
| Tensor | 目的Tensor类型，由函数实参自动推导。Tensor必须位于UB。 |
| Coord | 目的Tensor的坐标类型，由函数实参自动推导。 |
| RegDataType | 源寄存器元素类型，由函数实参自动推导。除`int4b_t`外，与目的Tensor的元素类型一致；目的Tensor为`int4b_t`时为`int4x2_t`。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输出 | 目的Tensor，必须位于UB。 |
| coord | 输入 | 搬出起始元素在`tensor`逻辑布局中的坐标。 |
| src | 输入 | 单源操作数重载的源寄存器。`direct`、`pack`和`pack_quarter`模式使用`src.mask`筛选有效元素；`store_1st`模式忽略`src.mask`。 |
| src0 | 输入 | `intlv`模式的第一个源寄存器。该模式忽略`src0.mask`。 |
| src1 | 输入 | `intlv`模式的第二个源寄存器。该模式忽略`src1.mask`。 |

<a id="表3-store_sideband_mode参数说明"></a>

**表3** `store_sideband_mode`参数说明

表中的VL表示一个矢量数据寄存器的长度，单位为Byte。`int4b_t`类型的数据在寄存器中以`int4x2_t`形式存储。`int4x2_t`数据类型由两个四位整型数打包为一个8比特存储单元，`fp4x2_e2m1_t`和`fp4x2_e1m2_t`数据类型由两个独立的四位浮点数打包为一个8比特存储单元，因此，对于这三种类型，下表所述的“元素”是指一个8比特对应的搬运单元，而不是单个4比特值。

| 取值 | 含义 | 搬出起始地址对齐约束（Byte） |
| --- | --- | --- |
| store_sideband_mode::direct | 正常模式，根据`src.mask`将源寄存器中的有效元素连续写入UB中的对应位置。 | 32 |
| store_sideband_mode::store_1st | 仅将`src`的第一个元素写入UB，忽略`src.mask`。 | `sizeof(RegDataType)` |
| store_sideband_mode::intlv | 交织模式，将`src0`和`src1`中的元素交替写入UB，得到2\*VL数据量，忽略两个源操作数的`mask`。仅支持双源操作数重载。 | 32 |
| store_sideband_mode::pack | 2倍压缩模式，根据`src.mask`筛选有效元素，将每个有效元素的低半部分bit写入压缩后的位置。 | min(32, VL/2) |
| store_sideband_mode::pack_quarter | 4倍压缩模式，根据`src.mask`筛选有效元素，将每个有效32bit元素的低8bit写入压缩后的位置。 | min(32, VL/4) |

## 数据类型

下表列出目的Tensor的元素类型。不同模式支持的数据类型如下，未列出的类型或模式组合会产生编译期错误。其中，源寄存器类型为`reg_tensor<int4x2_t>`时，目的Tensor对应的数据类型为`int4b_t`。

| sideband_mode | 支持的数据类型 |
| --- | --- |
| direct、store_1st | `int4b_t`、`int8_t`、`uint8_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t`、`hifloat8_t`、`fp8_e8m0_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float` |
| intlv | `int4b_t`、`int8_t`、`uint8_t`、`fp4x2_e2m1_t`、`fp4x2_e1m2_t`、`hifloat8_t`、`fp8_e8m0_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t` |
| pack | `int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`、`int64_t` |
| pack_quarter | `int32_t`、`uint32_t`、`float` |

## 返回值说明

无。

## 约束说明

- 仅支持位于UB的Tensor。
- 搬出起始地址必须满足[表3](#表3-store_sideband_mode参数说明)中的对齐约束。
- 传入`coord`时，坐标维数必须与Tensor布局匹配，且坐标不能越界。
- 单源操作数重载不支持`store_sideband_mode::intlv`；双源操作数重载仅支持`store_sideband_mode::intlv`。
- 使用`direct`、`pack`或`pack_quarter`模式前，必须保证`src.reg`和`src.mask`均已初始化。使用`store_1st`或`intlv`模式时，仅要求参与搬出的`reg`成员已初始化。
- 正常模式下，如果目的UB的剩余空间不足一个VL，必须通过`src.mask`关闭越界位置，并保证目的UB至少剩余一个DataBlock（32B）空间。
- 寄存器数据搬出应在`__simd_vf__`函数或`__simd_callee__`函数中完成。

## 调用示例

### 正常搬出

以下示例将源Tensor搬入寄存器，绑定全有效mask后写入目的Tensor。

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/arch/vector/mask_reg_compute.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

template <typename T, typename SrcTensor, typename DstTensor>
__simd_vf__ inline void LoadAndStore(const SrcTensor& src, DstTensor& dst)
{
    auto coord = asc::te::make_coord(0);
    auto value = asc::te::experimental::load(src, coord);
    value.with_mask(asc::te::experimental::all_mask<T>());
    asc::te::experimental::store(dst, coord, value);
}
```

### 交织搬出

```cpp
template <typename T, typename DstTensor>
__simd_vf__ inline void StoreInterleaved(
    DstTensor& dst,
    const asc::te::experimental::reg_tensor<T>& even,
    const asc::te::experimental::reg_tensor<T>& odd)
{
    asc::te::experimental::store<asc::te::experimental::store_sideband_mode::intlv>(
        dst, asc::te::make_coord(0), even, odd);
}
```
