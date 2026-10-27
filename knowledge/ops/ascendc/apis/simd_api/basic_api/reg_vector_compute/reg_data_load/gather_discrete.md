---
type: API
title: Gather
description: 根据索引值index将源操作数按元素收集到目的操作数dstReg中，index中每个元素表示相对于baseAddr的元素位置，支持多种数据类型组合。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/Gather_discrete.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h
---

# 离散搬入（Gather）<a name="ZH-CN_TOPIC_0000001956119717"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"`。

该指令会根据索引值index将源操作数按元素收集到目的操作数dstReg中。收集过程如图1所示：

**图 1**  Gather功能说明

![图1 Gather功能说明](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_gather_ub2reg.png)  


## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T0 = DefaultType, typename T1, typename T2 = DefaultType, typename T3, typename T4>
__simd_callee__ inline void Gather(T3& dstReg, __ubuf__ T1* baseAddr, T4& index, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
|-----|-----|
| T0 | 目的操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| T1 | 源操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| T2 | 索引值的数据类型，支持的数据类型请参考[数据类型](#数据类型)。 |
| T3 | 目的操作数的RegTensor类型。例如RegTensor\<half>，由编译器自动推导，用户不需要手动填写。 |
| T4 | 索引值的RegTensor类型，例如RegTensor\<uint16_t>，由编译器自动推导，用户不需要手动填写。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|-----|-----|-----|
| dstReg | 输出 | 目的操作数，类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| baseAddr | 输入 | 源操作数，Unified Buffer（UB）中的基地址，需要32字节对齐。 |
| index | 输入 | 索引值，dstReg中的每个元素在UB中相对于baseAddr的位置，单位：元素。类型为[RegTensor](../register_data_types/reg_tensor.md)。index中的值可以重复。<br>例：baseAddr：[elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7, ...]。 <br>每个元素相对于baseAddr的索引位置为：[0, 1, 2, 3, 4, 5, 6, 7, ...]。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

**表 3**  Gather操作数数据类型对应表

| 目的操作数 | 源操作数 | 索引值 |
|-----|-----|-----|
| int16_t | int8_t | uint16_t |
| int16_t | int16_t | uint16_t |
| uint16_t | uint8_t | uint16_t |
| uint16_t | uint16_t | uint16_t |
| half | half | uint16_t |
| bfloat16_t | bfloat16_t | uint16_t |
| int32_t | int32_t | uint32_t |
| uint32_t | uint32_t | uint32_t |
| float | float | uint32_t |
| int64_t | int64_t | uint32_t |
| int64_t | int64_t | uint64_t |
| uint64_t | uint64_t | uint32_t |
| uint64_t | uint64_t | uint64_t |

## 返回值说明

无

## 约束说明<a name="section177921451558"></a>

- 位于UB的地址必须32字节对齐。
- index索引值对应的数据必须在UB有效地址范围内。
- 当T0为b16类型，T1为b8数据类型时，目的操作数的低8位与源操作数相同，高8位自动补0。例如源操作数T1数据类型为int8_t：

    40=0b00101000 -> 0b0000000000101000，扩充至16位后等于40；

    -40=0b11011000 -> 0b0000000011011000，扩充至16位后等于216。

- 当源操作数数据类型T1为B64时，T0、T1、T2、T3、T4只支持以下组合：

    | T0 | T1 | T2 | T3（自动推导） | T4（自动推导） | 备注 |
    |-----|-----|-----|-----|-----|-----|
    | int64_t | int64_t | uint32_t | RegTensor<int64_t, RegTraitNumOne> | RegTensor<uint32_t> | index的前32个数有效 |
    | int64_t | int64_t | uint32_t | RegTensor<int64_t, RegTraitNumTwo> | RegTensor<uint32_t> | - |
    | int64_t | int64_t | uint64_t | RegTensor<int64_t, RegTraitNumOne> | RegTensor<uint64_t, RegTraitNumOne> | - |
    | int64_t | int64_t | uint64_t | RegTensor<int64_t, RegTraitNumOne> | RegTensor<uint64_t, RegTraitNumTwo> | index的前32个数有效 |
    | int64_t | int64_t | uint64_t | RegTensor<int64_t, RegTraitNumTwo> | RegTensor<uint64_t, RegTraitNumTwo> | - |
    | uint64_t | uint64_t | uint32_t | RegTensor<uint64_t, RegTraitNumOne> | RegTensor<uint32_t> | index的前32个数有效 |
    | uint64_t | uint64_t | uint32_t | RegTensor<uint64_t, RegTraitNumTwo> | RegTensor<uint32_t> | - |
    | uint64_t | uint64_t | uint64_t | RegTensor<uint64_t, RegTraitNumOne> | RegTensor<uint64_t, RegTraitNumOne> | - |
    | uint64_t | uint64_t | uint64_t | RegTensor<uint64_t, RegTraitNumOne> | RegTensor<uint64_t, RegTraitNumTwo> | index的前32个数有效 |
    | uint64_t | uint64_t | uint64_t | RegTensor<uint64_t, RegTraitNumTwo> | RegTensor<uint64_t, RegTraitNumTwo> | - |

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template <typename T, typename U>
__simd_vf__ inline void GatherVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, __ubuf__ U* indexAddr, uint32_t count, uint16_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::RegTensor<U> indexReg;
    AscendC::Reg::MaskReg mask;
    uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(indexReg, indexAddr + i * oneRepeatSize);
        AscendC::Reg::Gather(dstReg, srcAddr, indexReg, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

