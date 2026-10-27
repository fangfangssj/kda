---
type: API
title: Maxs
description: 根据mask对源操作数srcReg按元素与标量比较取较大值，将结果写入目的操作数dstReg。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Maxs.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h
---

# Maxs<a name="ZH-CN_TOPIC_0000001929827668"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h"`。

该接口根据mask，对源操作数srcReg按元素与标量相比，如果比标量大，则取源操作数值；如果比标量小，则取标量值，将结果写入目的操作数dstReg。计算公式如下：

$$
dstReg_i = \max(srcReg_i, scalarValue)
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Maxs(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 矢量目的操作数和源操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 标量源操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>&bull; MERGING模式当前不支持。 |
| S | dstReg为RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| scalarValue | 输入 | 源操作数。<br>类型为标量。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

输入srcReg为-0，scalarValue为+0的情况下，输出dstReg为+0。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void MaxsVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, T scalarValue, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;    
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::Maxs(dstReg, srcReg, scalarValue, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [SimThreadExponential 950 ACLNN 算子知识](../../../../../operators/aclnn/math/random/sim_thread_exponential_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [RmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_dynamic_mx_quant_950.md)
- [AvgPool2d 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/pooling/avg_pool2d_950.md)
- [DequantSwigluQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant_950.md)
- [DynamicBlockMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_block_mx_quant_950.md)
- [DynamicDualLevelMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_dual_level_mx_quant_950.md)
- [DynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_mx_quant_950.md)
- [DynamicMxQuantWithDualAxis 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_mx_quant_with_dual_axis_950.md)
