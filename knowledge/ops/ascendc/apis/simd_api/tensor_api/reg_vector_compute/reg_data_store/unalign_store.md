---
type: API
title: unalign_store
description: 头文件路径为："tensor_api/experimental/arch/vector/reg_data_store.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_data_store/unalign_store.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/experimental/arch/vector/reg_data_store.h
- include/tensor_api/tensor.h
- include/tensor_api/experimental/vector_ops.h
---

# unalign_store

## 功能说明

头文件路径为：`"tensor_api/experimental/arch/vector/reg_data_store.h"`，命名空间为`asc::te::experimental`。

`unalign_store`用于将[reg_tensor](../reg_tensor/reg_tensor.md)中的数据连续搬出到非32字节对齐的Unified Buffer（UB）地址。对象内部维护一个非对齐搬出寄存器，提供以下两个操作：

- `store`：将源寄存器前`count`个元素连续搬出。已形成的32字节对齐主块写入UB，非对齐尾块暂存到对象内部的非对齐寄存器中，供下一次连续搬出拼接。
- `flush`：将非对齐寄存器中暂存的尾块写入UB。一次连续搬出序列结束后必须调用该接口完成收尾。

`store`使用`src.reg`，忽略`src.mask`。有效搬出元素数由`count`指定，不能使用mask替代`count`。本接口不会自动推进目的Tensor或坐标，调用者须根据累计搬出元素数指定下一次连续写入地址。

## 函数原型

```cpp
struct unalign_store {
    __simd_callee__ unalign_store() = default;

    template <typename Tensor, typename RegDataType>
    __simd_callee__ inline void store(
        Tensor dst_tile, const reg_tensor<RegDataType>& src, uint32_t count);

    template <typename Tensor, typename Coord, typename RegDataType>
    __simd_callee__ inline void store(
        Tensor dst_tile, const Coord& coord,
        const reg_tensor<RegDataType>& src, uint32_t count);

    template <typename Tensor>
    __simd_callee__ inline void flush(Tensor dst_base, int32_t coord);
};
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| Tensor | 目的Tensor类型，由函数实参自动推导，必须位于UB。 |
| Coord | `store`的目的Tensor坐标类型，由函数实参自动推导。 |
| RegDataType | 源寄存器元素类型，由函数实参自动推导。除`int4b_t`映射为`int4x2_t`外，必须与目的Tensor元素类型一致。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst_tile | 输出 | 本次搬出的目的Tensor。不传入坐标时，写入地址为`dst_tile.engine().begin().get()`。Tensor视图按值传入，不会修改调用者的视图，但会写入其指向的UB。 |
| coord（store） | 输入 | 本次搬出起始元素在`dst_tile`逻辑布局中的坐标，实际写入地址为`(dst_tile.data() + dst_tile.layout()(coord)).get()`。 |
| src | 输入 | 源`reg_tensor`，`reg`必须初始化，`mask`不参与搬出。 |
| count | 输入 | 本次搬出的元素个数，类型为`uint32_t`，不超过单个矢量寄存器可容纳的元素个数。 |
| dst_base | 输出 | 收尾使用的目的Tensor，其起始地址为本次连续搬出序列第一次`store`的实际写入地址。 |
| coord（flush） | 输入 | 类型为`int32_t`，传给底层收尾接口的元素偏移量，应为本次连续搬出序列各次`count`之和。它不是`make_coord`生成的逻辑坐标，也不是字节偏移。 |

`flush`使用的收尾地址由`dst_base.engine().begin().get()`和元素偏移`coord`确定。若第一次`store`指定了非零坐标，应将该坐标对应的Tensor视图作为`dst_base`，不能直接用未偏移的原Tensor配合本序列元素总数进行收尾。

## 数据类型

| 目的Tensor元素类型 | 源寄存器类型 |
| --- | --- |
| `int4b_t` | `reg_tensor<int4x2_t>`。 |
| `fp4x2_e2m1_t`、`fp4x2_e1m2_t` | 对应元素类型的`reg_tensor`。 |
| `int8_t`、`uint8_t`、`hifloat8_t`、`fp8_e8m0_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t` | 对应元素类型的`reg_tensor`。 |
| `int16_t`、`uint16_t`、`half`、`bfloat16_t` | 对应元素类型的`reg_tensor`。 |
| `int32_t`、`uint32_t`、`float` | 对应元素类型的`reg_tensor`。 |
| `int64_t` | `reg_tensor<int64_t>`。 |

## 返回值说明

`store`和`flush`均无返回值。

## 约束说明

- 目的地址无需32字节对齐，但必须满足元素类型的对齐要求，实际写入范围不能超出有效UB空间。
- 同一连续搬出序列必须复用同一个`unalign_store`对象，所有目的Tensor的元素类型必须一致。
- 相邻两次`store`的实际写入地址必须连续，即下一次起始地址为上一次起始地址加已搬出数据长度。地址不连续时，应先对当前序列调用`flush`，再开始新的搬出序列。
- 每个序列的全部`store`完成后，使用首次写入地址对应的`dst_base`以及累计搬出元素数调用一次`flush`。不能遗漏收尾，也不能在尚未进行`store`时调用`flush`。
- `count`不得超过一个寄存器容量；例如`uint8_t`为`VL / sizeof(uint8_t)`个元素，`float`为`VL / sizeof(float)`个元素。
- `store`的坐标维数必须与Tensor布局匹配，坐标不能越界；`flush`的元素偏移及最终地址必须在合法范围内。

## 调用示例

以下示例连续复制`uint8_t`数据，最后一段通过`count`控制有效搬出数量。源、目的Tensor均为一维UB Tensor，起始地址可非32字节对齐；`total_elements`大于0且不超过`int32_t`上限。源UB空间必须为最后一次完整VL读取及前置缓存预留合法空间，目的UB空间必须覆盖全部写入数据。

```cpp
#include "tensor_api/tensor.h"
#include "tensor_api/experimental/vector_ops.h"

template <typename SrcTensor, typename DstTensor>
__simd_vf__ inline void CopyUnaligned(
    SrcTensor src, DstTensor dst, uint32_t total_elements)
{
    constexpr uint32_t elements_per_register = asc_get_vf_len() / sizeof(uint8_t);
    asc::te::experimental::unalign_load loader;
    asc::te::experimental::unalign_store storer;
    loader.prepare(src);
    for (uint32_t offset = 0; offset < total_elements; offset += elements_per_register) {
        uint32_t remaining = total_elements - offset;
        uint32_t count = remaining < elements_per_register ? remaining : elements_per_register;
        auto coord = asc::te::make_coord(offset);
        auto value = loader.load(src, coord);
        storer.store(dst, coord, value, count);
    }
    storer.flush(dst, static_cast<int32_t>(total_elements));
}
```
