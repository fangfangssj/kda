---
type: API
title: copy（Unified Buffer到Global Memory的数据搬运）
description: 头文件路径为：tensor_api/tensor.h，命名空间为asc::te。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/vector_compute_store/copy_ub_to_gm.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# copy（Unified Buffer到Global Memory的数据搬运）

> [!NOTE]说明
> 本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 功能说明

头文件路径为：`tensor_api/algorithm/copy.h`，命名空间为`asc::te`。

该接口用于将Unified Buffer（UB）中的数据搬运到Global Memory。接口根据源、目的Tensor的存储位置自动选择`copy_ub_to_gm`通路，并根据源、目的Layout推导搬运块数、单块长度、源步长和目的步长，支持单块长度非32字节对齐的数据搬运。

当前支持ND2ND、DN2DN、NZ2NZ和ZN2ZN搬运。搬运过程中不执行数据类型转换，数据格式和有效内容保持不变。接口支持整Tensor搬运和通过坐标、Shape指定的局部区域搬运。接口使用Global Memory目的Tensor设置的[L2 Cache Hint](../../tensor/tensor.md#set_l2_cache_hint)确定本次搬运的L2 Cache模式。

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

UB到Global Memory通路使用`copy_ub_to_gm{}`作为搬运操作对象，默认trait为`ub_to_gm_trait_default{}`。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| atom | 输入 | 搬运原子对象，可通过`make_copy(copy_ub_to_gm{})`或`make_copy(copy_ub_to_gm{}, ub_to_gm_trait_default{})`构造。 |
| dst | 输出 | 目的Tensor，存储位置必须为`location::gm`。本次搬运的L2 Cache模式从目的Tensor的L2 Cache Hint中获取，设置方法参见[L2 Cache模式](#l2-cache模式)。 |
| src | 输入 | 源Tensor，存储位置必须为`location::ub`。 |
| dst_coord | 输入 | 目的区域起始坐标。显式坐标的结构应与`copy_shape`一致，也可以使用`zero_coord`。 |
| src_coord | 输入 | 源区域起始坐标。显式坐标的结构应与`copy_shape`一致，也可以使用`zero_coord`。 |
| copy_shape | 输入 | 搬运区域的逻辑Shape。接口根据该Shape及源、目的Layout推导起始地址和搬运参数。 |

### 搬运参数推导

`copy`根据源、目的Tensor的Layout自动推导`blockCount`、`blockLen`、源步长和目的步长。Tensor API中的单块长度和步长均以字节为单位，步长表示相邻数据块起始地址的间隔。

对于整Tensor搬运，各Layout模式的推导规则如下：

| Layout模式 | 搬运块数 | 单块长度 | 源、目的步长 |
| :--- | :--- | :--- | :--- |
| ND | 源Layout的总行数 | 源、目的Layout总列字节数的较小值 | 对应Tensor的行步长 × 元素字节数 |
| DN | 源Layout的总列数 | 源、目的Layout总行字节数的较小值 | 对应Tensor的列步长 × 元素字节数 |
| NZ | 分形格式的外层列数 | 总行数 × 内层列数 × 元素字节数 | 对应Tensor的外层列步长 × 元素字节数 |
| ZN | 分形格式的外层行数 | 总列数 × 内层行数 × 元素字节数 | 对应Tensor的外层行步长 × 元素字节数 |

## 数据类型

源Tensor和目的Tensor的数据类型必须相同。

| 源/目的Layout | 支持的数据类型 |
| :--- | :--- |
| ND、DN、NZ、ZN | `int4b_t`、`fp4x2_e1m2_t`、`fp4x2_e2m1_t`、`int8_t`、`uint8_t`、`hifloat8_t`、`fp8_e5m2_t`、`fp8_e4m3fn_t`、`int16_t`、`uint16_t`、`half`、`bfloat16_t`、`int32_t`、`uint32_t`、`float`、`int64_t` |

## 返回值说明

`copy`无返回值。`make_copy`返回`copy_atom`对象。

## 约束说明

- 源Tensor必须位于UB，目的Tensor必须位于Global Memory。
- 源UB地址需满足32字节对齐要求，目的Global Memory地址支持1字节对齐。
- 源、目的Layout Pattern必须相同。当前支持ND2ND、DN2DN和NZ2NZ、ZN2ZN。
- 源、目的数据类型必须相同，目的Tensor或目的区域容量必须不小于搬运区域。
- Layout推导出的搬运块数、单块长度和步长必须满足底层`asc_copy_ub2gm_align`指令的字段范围及对齐要求。有效搬运的块数取值范围为[1, 4095]，单块长度取值范围为[1, 2097151]字节，且单块长度必须为元素字节数的整数倍。
- 有效搬运的源步长必须等于单块长度，或为32字节的正整数倍。前者表示UB中的数据块紧凑排列，后者表示各数据块按32字节边界排列。
- 与部分产品上的`DataCopyExtParams`接口不同，Tensor API要求源、目的Layout以及区域搬运Shape合法，不能依赖搬运块数或单块长度为0表示空操作。
- b4类型按b8搬运粒度调整块长度和步长。

## 关键特性说明

### 非对齐搬出

UB源地址按32字节对齐，Global Memory目的端允许1字节对齐。单块长度为32字节的整数倍时，块内数据全部写入Global Memory；单块长度不是32字节的整数倍时，底层指令按32字节粒度处理UB侧尾部，但只将单块长度指定的有效字节写入Global Memory，尾部补齐数据不会写入目的Tensor。

以元素类型为`half`、搬运Shape为`(2, 23)`的ND Tensor为例，单块有效长度为`23 × sizeof(half) = 46`字节：

| UB源Tensor行步长 | GM目的Tensor行步长 | 推导结果 | 搬运方式 |
| :--- | :--- | :--- | :--- |
| 32个元素（64字节） | 23个元素（46字节） | 块数为2、单块长度为46字节、源步长为64字节、目的步长为46字节 | UB侧每块按32字节边界排列，尾部18字节不写入GM；GM侧有效数据连续排列。 |
| 23个元素（46字节） | 23个元素（46字节） | 块数为2、单块长度为46字节、源步长和目的步长均为46字节 | UB和GM中的有效数据块均紧凑排列。 |

当目的步长大于单块长度时，Global Memory中相邻有效数据块之间保留`目的步长 - 单块长度`字节的间隔，该间隔不会被写入。

### L2 Cache模式

`copy`从位于Global Memory的目的Tensor中获取L2 Cache Hint，作为本次搬运的L2 Cache模式，无需向`copy`额外传参。

调用`copy`前，可通过目的Tensor的[set_l2_cache_hint()](../../tensor/tensor.md#set_l2_cache_hint)接口设置L2 Cache Hint，通过[get_cache_mode()](../../tensor/tensor.md#get_cache_mode)接口获取当前配置。未显式设置时，默认值为`asc::te::cache_mode::normal`。

```cpp
dst.set_l2_cache_hint(asc::te::cache_mode::normal);
auto cache_mode = dst.get_cache_mode();
asc::te::copy(dst, src);
```

## 调用示例

```cpp
#include "tensor_api/tensor.h"

constexpr uint32_t M = 8;
constexpr uint32_t N = 64;

__aicore__ inline void copy_ub_to_gm_example(__gm__ half* gm_addr)
{
    __ubuf__ half ub_buf[M * N];

    auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn,
        asc::te::layout_trait_default<half>>(M, N);
    auto src = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub>(ub_buf), layout);
    auto dst = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gm_addr), layout);

    asc::te::copy(dst, src);
}
```

该示例的ND Layout包含8行，每行64个`half`元素，因此接口推导出块数为8、单块长度为128字节，源步长和目的步长均为128字节。单块长度满足32字节对齐要求，UB和GM中的数据块均连续排列。

以下示例显式构造搬运原子对象，并搬运单块长度非32字节对齐的局部区域：

```cpp
auto atom = asc::te::make_copy(asc::te::copy_ub_to_gm{}, asc::te::ub_to_gm_trait_default{});
asc::te::copy(atom, dst, src, asc::te::make_coord(1, 0), asc::te::make_coord(2, 0),
    asc::te::make_shape(2, 23));
```

该区域包含2行，每行搬运23个`half`元素，因此接口推导出块数为2、单块长度为46字节。源、目的Tensor的行步长均为128字节，所以源步长和目的步长均为128字节。每块仅有46字节有效数据写入GM，UB侧尾部补齐数据不会写入；GM中相邻有效数据块之间保留82字节间隔。

## 配套 API 与常见组合

- [copy（Global Memory到Unified Buffer的数据搬运）](../vector_compute_load/copy_gm_to_ub.md)
- [make_tensor](../../tensor/make_tensor.md)
- [make_mem_ptr](../../tensor/make_mem_ptr.md)
- [make_frame_layout](../../layout/make_frame_layout.md)
- [Algorithm](../../algorithm.md)
- [对齐单位、布局行距与填充值](../../../../../guides/memory/alignment_units_layout_stride_and_padding_values.md)
