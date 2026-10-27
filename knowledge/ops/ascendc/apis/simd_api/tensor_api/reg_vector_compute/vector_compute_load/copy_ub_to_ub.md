---
type: API
title: copy（Unified Buffer到Unified Buffer的数据搬运）
description: 头文件路径为：tensor_api/tensor.h，命名空间为asc::te。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/vector_compute_load/copy_ub_to_ub.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# copy（Unified Buffer到Unified Buffer的数据搬运）

> [!NOTE]说明
> 本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 功能说明

头文件路径为：`tensor_api/algorithm/copy.h`，命名空间为`asc::te`。

该接口用于在Unified Buffer（UB）内部进行高维切分数据搬运。接口根据源、目的Tensor的存储位置自动选择`copy_ub_to_ub`通路，并根据Layout推导数据块个数、单块长度以及源、目的块间间隔，支持连续和非连续搬运。

支持ND2ND、DN2DN、NZ2NZ和ZN2ZN搬运。源、目的Layout Pattern必须相同，搬运过程中不执行格式转换和数据类型转换，数据格式和内容保持不变。该接口适合调整UB中的数据存放位置，以及在带步长的Tensor之间搬运局部区域。

## 函数原型

```cpp
template <typename DstTensor, typename SrcTensor>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src)

template <typename Atom, typename DstTensor, typename SrcTensor>
__aicore__ inline void copy(const copy_atom<Atom>& atom,
    const DstTensor& dst, const SrcTensor& src)

template <typename DstTensor, typename SrcTensor, typename DstCoord,
    typename SrcCoord, typename CopyShape>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src,
    const DstCoord& dst_coord, const SrcCoord& src_coord,
    const CopyShape& copy_shape)

template <typename CopyOperation>
__aicore__ inline constexpr auto make_copy(const CopyOperation& operation)

template <typename CopyOperation, typename CopyTrait>
__aicore__ inline constexpr auto make_copy(
    const CopyOperation& operation, const CopyTrait& trait)
```

UB到UB通路使用`copy_ub_to_ub{}`作为搬运操作对象，默认trait为`ub_to_ub_trait_default{}`。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| atom | 输入 | 搬运原子对象，可通过`make_copy(copy_ub_to_ub{})`或`make_copy(copy_ub_to_ub{}, ub_to_ub_trait_default{})`构造。 |
| dst | 输出 | 目的Tensor，存储位置必须为`location::ub`。 |
| src | 输入 | 源Tensor，存储位置必须为`location::ub`。 |
| dst_coord | 输入 | 目的区域起始坐标。显式坐标的结构应与`copy_shape`一致，也可以使用`zero_coord`。 |
| src_coord | 输入 | 源区域起始坐标。显式坐标的结构应与`copy_shape`一致，也可以使用`zero_coord`。 |
| copy_shape | 输入 | 搬运区域的逻辑Shape。接口根据该Shape及源、目的Layout推导起始地址和搬运参数。 |

### 参数推导

`copy`根据源、目的Tensor的Layout自动推导[asc_copy_ub2ub高维切分数据搬运](../../../c_api/vector_datamove/asc_copy_ub2ub.md#参数说明)所需的搬运参数，无需手动传入。对于整Tensor搬运，各Layout模式的推导规则如下。

| Layout模式 | 数据块个数 | 单块长度（字节） | 源、目的块间间隔（字节） |
| :--- | :--- | :--- | :--- |
| ND | Layout的总行数 | Layout的总列数 × 元素字节数 | 对应Tensor的行步长 × 元素字节数 - 单块长度 |
| DN | Layout的总列数 | Layout的总行数 × 元素字节数 | 对应Tensor的列步长 × 元素字节数 - 单块长度 |
| NZ | 分形格式的外层列数 | 总行数 × 内层列数 × 元素字节数 | 对应Tensor的外层列步长 × 元素字节数 - 单块长度 |
| ZN | 分形格式的外层行数 | 总列数 × 内层行数 × 元素字节数 | 对应Tensor的外层行步长 × 元素字节数 - 单块长度 |

推导结果与`asc_copy_ub2ub`高维切分数据搬运参数的对应关系如下：

| 推导结果 | 对应参数 | 换算关系 |
| :--- | :--- | :--- |
| 数据块个数 | `burst_count` | 无需换算。 |
| 单块长度（字节） | `burst_len` | 单块长度除以32，单位为DataBlock（32字节）。 |
| 源块间间隔（字节） | `src_gap` | 源块间间隔除以32，单位为DataBlock（32字节）。 |
| 目的块间间隔（字节） | `dst_gap` | 目的块间间隔除以32，单位为DataBlock（32字节）。 |

使用区域搬运重载时，接口根据`src_coord`、`dst_coord`和`copy_shape`解析起始地址和搬运区域，并根据区域Shape推导数据块个数和单块长度；块间间隔仍由源、目的Layout决定。

## 数据类型

源Tensor和目的Tensor的数据类型必须相同。

| 源/目的Layout | 支持的数据类型 |
| :--- | :--- |
| ND、DN、NZ、ZN | `int4b_t`、`fp4x2_e1m2_t`、`fp4x2_e2m1_t`、`int8_t`、`uint8_t`、`hifloat8_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`、`int64_t` |

## 返回值说明

`copy`无返回值。`make_copy`返回`copy_atom`对象。

## 约束说明

- 源Tensor和目的Tensor均必须位于UB，源、目的地址均需满足32字节对齐要求。
- 源、目的Layout Pattern必须相同。支持ND2ND、DN2DN、NZ2NZ和ZN2ZN。
- 源、目的数据类型必须相同，目的Tensor或目的区域容量必须不小于搬运区域。
- 单块搬运长度和源、目的块间间隔必须为32字节的整数倍，Layout推导结果必须满足该对齐要求。
- Layout推导出的数据块个数取值范围为[1, 4095]；单块长度取值范围为[1, 65535]个DataBlock。源、目的块间间隔均使用16位字段表示，取值范围为[0, 65535]个DataBlock。
- 与部分产品上的`DataCopyParams`接口不同，Tensor API要求有效Layout推导出的数据块个数和单块长度均不为0。
- b4类型按b8搬运粒度调整块长度和块间间隔。

## 关键特性说明

### 整Tensor和局部区域搬运

`copy(dst, src)`根据源Layout描述的完整数据范围生成搬运参数。区域重载只搬运`copy_shape`指定的区域，并分别使用`src_coord`和`dst_coord`计算起始地址。

### 连续和非连续搬运

当源、目的块间间隔均为0时，相邻数据块首尾相接，属于连续搬运。当源或目的块间间隔大于0时，属于非连续搬运；接口只搬运数据块覆盖的区域，不读取源间隔，也不写入目的间隔。当数据块个数为1时，块间间隔不影响搬运结果。

以元素类型为`int8_t`、搬运Shape为`(2, 32)`的ND Tensor为例，单块长度为32字节，即1个DataBlock：

| 场景 | 源Tensor行步长 | 目的Tensor行步长 | 推导结果 |
| :--- | :--- | :--- | :--- |
| 连续搬运 | 32个元素 | 32个元素 | `blockCount = 2`、`blockLen = 1`、`srcGap = 0`、`dstGap = 0`。 |
| 非连续搬运 | 64个元素 | 96个元素 | `blockCount = 2`、`blockLen = 1`、`srcGap = 1`、`dstGap = 2`。 |

上述参数仅用于说明Layout推导结果，调用`copy`时无需构造或传入`DataCopyParams`。

### ZN2ZN分形搬运

ZN Layout将逻辑二维Shape拆分为`((C0, ceil_div(M, C0)), (16, ceil_div(N, 16)))`。其中，`C0`由元素类型决定。ZN2ZN整Tensor搬运以外层行方向切分数据块：

- `blockCount`为外层行数`ceil_div(M, C0)`。
- `blockLen`为逻辑总列数乘以内层行数和元素字节数，即`N × C0 × sizeof(T)`字节。
- `srcGap`和`dstGap`分别为对应Tensor的外层行步长减去`blockLen`，并换算为32字节的DataBlock数。

源Tensor和目的Tensor必须均为ZN Layout，接口只在相同分形格式之间搬运，不改变分形排布。

## 调用示例

以下示例在UB内完成数据搬运：源与目的张量位于同一UB地址空间，搬运参数（blockCount、blockLen、srcGap、dstGap）由layout的shape与stride自动推导，无需显式配置。完整Tensor搬运形式如下：

```cpp
#include "tensor_api/tensor.h"

constexpr uint32_t M = 8;
constexpr uint32_t N = 64;

__aicore__ inline void copy_ub_to_ub_example()
{
    __ubuf__ int8_t src_buf[M * N];
    __ubuf__ int8_t dst_buf[M * N];

    auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
        asc::te::layout_trait_default<int8_t>>(M, N);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(src_buf), layout);
    auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(dst_buf), layout);

    asc::te::copy(dst, src);
}
```

该示例的ND Layout包含8行，每行64个`int8_t`元素，因此接口推导出`blockCount = 8`、`blockLen = 2`、`srcGap = 0`、`dstGap = 0`。

局部区域搬运示例如下：

```cpp
asc::te::copy(dst, src, asc::te::make_coord(0, 0), asc::te::make_coord(2, 0),
    asc::te::make_shape(4, 32));
```

该区域包含4行，每行搬运32个`int8_t`元素。源、目的Tensor的行步长均为64字节，因此接口推导出`blockCount = 4`、`blockLen = 1`、`srcGap = 1`、`dstGap = 1`。

ZN2ZN分形搬运示例如下：

```cpp
#include "tensor_api/tensor.h"

constexpr uint32_t ZN_M = 32;
constexpr uint32_t ZN_N = 32;

__aicore__ inline void copy_ub_to_ub_zn_example()
{
    __ubuf__ half src_buf[ZN_M * ZN_N];
    __ubuf__ half dst_buf[ZN_M * ZN_N];

    auto layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn,
        asc::te::layout_trait_default<half>>(ZN_M, ZN_N);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(src_buf), layout);
    auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(dst_buf), layout);

    asc::te::copy(dst, src);
}
```

对于`half`，`C0`为16。该示例的ZN Layout为`((16, 2), (16, 2))`，接口沿外层行方向生成2个数据块；每个数据块包含32列和16个内层行元素，共1024字节，即32个DataBlock。源、目的Layout均为紧凑排布，因此`srcGap = 0`、`dstGap = 0`。
