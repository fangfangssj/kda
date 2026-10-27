---
type: API
title: Gather
description: 根据索引值将源操作数按元素收集到目的操作数中，支持b8、b16、b32数据类型。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/scatter_operations/Gather.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_gather_mask_intf.h
---

# Gather<a name="ZH-CN_TOPIC_0000002095379710"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_gather_mask_intf.h"`。

该指令会根据索引值indexReg将源操作数srcReg按元素收集到目的操作数dstReg中。收集过程如下图所示：

**图 1**  Gather功能说明  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_gather_1.png "Gather功能说明")

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void Gather(S& dstReg, S& srcReg, V& indexReg)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 源操作数和目的操作数的数据类型，两者需保持一致。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 索引值的数据类型，支持的数据类型请参考[数据类型](#数据类型)。 |
| S | 目的操作数的[RegTensor](../register_data_types/reg_tensor.md)类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要手动填写。 |
| V | 索引值的[RegTensor](../register_data_types/reg_tensor.md)类型，例如RegTensor&lt;uint16_t&gt;，由编译器自动推导，用户不需要手动填写。 |

**表 2**  函数参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| indexReg | 输入 | 索引值，单位：元素。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>数据类型的位宽需要与目的操作数的位宽保持一致。indexReg中的值可以重复。<br>如果indexReg中索引值超出当前RegTensor中能存储的最大数据元素个数时，按照如下方式处理：<br>RegTensor所能存储的最大数据元素个数为[VL](../register_data_types/reg_tensor.md)/sizeof(T)，indexReg中索引值为i，索引值更新为i % (VL/sizeof(T))。<br>例：元素为half数据类型的srcReg，有128个元素，当indexReg中索引值为168时，索引值更新为168 % 128 = 40。 |

## 数据类型


- Ascend 950PR&950DT系列产品，源操作数和目的操作数支持的数据类型为：b8、b16、b32。
- Ascend 950PR&950DT系列产品，索引值支持的数据类型为：uint8_t、uint16_t、uint32_t。



## 返回值说明

无

## 约束说明<a name="section177921451558"></a>

- 源操作数和目的操作数的数据类型需保持一致。
- 索引值和目的操作数的数据类型位宽需保持一致。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 根据索引从源向量中收集元素，dstReg[i] = srcReg[indexReg[i]]
template <typename T, typename U>
__simd_vf__ inline void GatherVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, __ubuf__ U* indexAddr, uint32_t count, uint16_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> srcReg, dstReg;
    AscendC::Reg::RegTensor<U> indexReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(srcReg, srcAddr, oneRepeatSize);
        AscendC::Reg::LoadAlign<U, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(indexReg, indexAddr, oneRepeatSize);
        AscendC::Reg::Gather<T, U>(dstReg, srcReg, indexReg);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, dstReg, oneRepeatSize, mask);
    }
}
```
