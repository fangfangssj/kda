---
type: API
title: Axpy
description: 对标量与源操作数执行乘加操作，将标量乘积加到目的操作数上。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.compute.composite
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/composite_compute/Axpy.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_ternary_scalar_intf.h
---

# Axpy<a name="ZH-CN_TOPIC_0000002159438184"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_vec_ternary_scalar_intf.h"`。

该接口根据mask对dstReg、srcReg和标量scalarValue按元素执行乘加操作，将结果写入dstReg。计算公式如下：

$$
dstReg_i = scalarValue \times srcReg_i + dstReg_i
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Axpy(S& dstReg, S& srcReg, const U scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :----- | :--- |
| T | 操作数数据类型。 |
| U | 标量源操作数的数据类型。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型。选择MERGING模式或ZEROING模式。<br>• ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>• MERGING模式当前不支持。 |
| S | 目的操作数的`RegTensor`类型，例如`RegTensor<half>`，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------- | :--- |
| dstReg | 输出 | 目的操作数和源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。数据类型需要与源操作数保持一致。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>数据类型需要与目的操作数保持一致。 |
| scalarValue | 输入 | 源操作数。<br>类型为标量。<br>数据类型需要与目的操作数保持一致。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 数据类型

**表3**  数据类型组合

| srcReg   | scalarValue | dstReg   |
| :------- | :---------- | :------- |
| half     | half        | half     |
| float    | float       | float    |
| int64_t  | int64_t     | int64_t  |
| uint64_t | uint64_t    | uint64_t |

## 约束说明<a name="section633mcpsimp"></a>

- 本接口支持寄存器重叠。
- 本接口操作数为寄存器，不涉及地址对齐。
- 本接口不修改全局寄存器的值。

## 调用示例<a name="section98571172212"></a>

```cpp
template<typename T, typename U>
static __simd_vf__ inline void AxpyVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, U scalarValue, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;    
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(dstReg, dstAddr + i * oneRepeatSize);
        AscendC::Reg::Axpy(dstReg, srcReg, scalarValue, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [DepthwiseConv2D 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/conv/depthwise_conv_2d_950.md)
- [ForeachLerpScalar 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_lerp_scalar_950.md)
- [Atan2 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/atan2_950.md)
- [GeluQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/gelu_quant_950.md)
- [Gelu 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/activation/gelu_950.md)
- [DynamicQuant 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/quant/dynamic_quant_950.md)
