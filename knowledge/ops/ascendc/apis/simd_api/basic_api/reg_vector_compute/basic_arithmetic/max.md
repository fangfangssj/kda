---
type: API
title: Max
description: 根据mask对源操作数srcReg0、srcReg1按元素求最大值，将结果写入目的操作数dstReg。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Max.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h
---

# Max<a name="ZH-CN_TOPIC_0000001956827097"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"`。

该接口根据mask，对源操作数srcReg0、srcReg1进行按元素求最大值操作，将结果写入目的操作数dstReg。计算公式如下：

$$
dstReg_i = \max(srcReg0_i, srcReg1_i)
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Max(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>&bull; MERGING模式下，mask未筛选的元素在dstReg中保留dstReg原值。 |
| U | 源操作数和目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg0 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。|
| srcReg1 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

输入srcReg0为-0，srcReg1为+0的情况下，输出dstReg为+0。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void MaxVf(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::Reg::Max(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [GeluQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/gelu_quant_950.md)
- [AttentionUpdate 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/attention_update_950.md)
- [SoftmaxCrossEntropyWithLogits 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/softmax_cross_entropy_with_logits_950.md)
- [RingAttentionUpdate 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/ring_attention_update_950.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [CrossEntropyLoss 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/loss/cross_entropy_loss_950.md)
- [AlltoAllQuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md)
- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
