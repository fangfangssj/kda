---
type: API
title: Neg
description: 根据mask对源操作数srcReg逐元素取相反数，将结果写入目的操作数dstReg。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Neg.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_unary_intf.h
---

# Neg<a name="ZH-CN_TOPIC_0000001956986913"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_unary_intf.h"`。

该接口用于根据mask对输入数据srcReg进行取相反数操作，将结果写入dstReg。计算公式如下：

$$
dstReg_i = -srcReg_i
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Neg(U& dstReg, U& srcReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>&bull; MERGING模式当前不支持。 |
| U | 源操作数和目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| mask | 输入 | 源操作数中元素有效性的指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：int8_t、int16_t、half、int32_t、float、int64_t。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void NegVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::Neg(dstReg, srcReg, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [InplaceNeg 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/inplace_neg_950.md)
- [SwishBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swish_backward_950.md)
