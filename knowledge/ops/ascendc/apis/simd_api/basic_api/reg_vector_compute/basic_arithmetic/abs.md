---
type: API
title: Abs
description: 对srcReg中有效元素逐个取绝对值并写入dstReg，复数类型取模。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Abs.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_unary_intf.h
---

# Abs<a name="ZH-CN_TOPIC_0000001956810269"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_unary_intf.h"`。

- 对实数类型

    对srcReg中的有效元素逐个取绝对值，并将结果写入dstReg对应位置，计算公式如下：

    $$
    dstReg_i = |srcReg_i|
    $$

- 对复数类型

    对srcReg中有效元素逐个取模，并将结果写入dstReg对应位置，计算公式如下：

    $$
    dstReg_i = |srcReg_i| = (\alpha^2 + \beta^2)^{1/2}
    $$

    其中$srcReg_i = \alpha + \beta i$，α为复数的实部，β为复数的虚部。

## 函数原型<a name="section620mcpsimp"></a>

- 实数计算

    ```cpp
    template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
    __simd_callee__ inline void Abs(U& dstReg, U& srcReg, MaskReg& mask)
    ```

- 复数计算

    ```cpp
    template <typename T = DefaultType, typename U = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S, typename V>
    __simd_callee__ inline void Abs(S& dstReg, V& srcReg, MaskReg& mask)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 目的操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 源操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>&bull; MERGING模式当前不支持。 |
| S | 目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |
| V | 源操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

**表 3**  数据类型组合

| srcReg | dstReg |
| --- | --- |
| int8_t | int8_t |
| int16_t | int16_t |
| half | half |
| int32_t | int32_t |
| float | float |
| complex32 | half |
| int64_t | int64_t |
| complex64 | float |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 当目的操作数和源操作数数据类型不一致时，目的操作数和源操作数不可重叠。
- 整型数据的计算结果如果超出数据类型的表示范围会采取非饱和截断，比如int8_t类型，srcReg为-128，其绝对值128会被截断成-128。

- 针对Ascend 950PR&950DT系列产品，仅complex32和complex64类型的接口内部计算涉及Subnormal，处理方式受编译选项--cce-ftz控制（默认值为true）：
    - --cce-ftz=false时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    - --cce-ftz=true时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例<a name="section642mcpsimp"></a>

- 实数计算

    ```cpp
    template<typename T>
    __simd_vf__ inline void AbsVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::Reg::RegTensor<T> srcReg;
        AscendC::Reg::RegTensor<T> dstReg;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::Reg::UpdateMask<T>(count);
            AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
            AscendC::Reg::Abs(dstReg, srcReg, mask);
            AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

- 复数计算

    ```cpp
    template<typename T, typename U>
    __simd_vf__ inline void AbsVF(__ubuf__ U* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::Reg::RegTensor<T> srcReg;
        AscendC::Reg::RegTensor<U> dstReg;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::Reg::UpdateMask<T>(count);
            AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
            AscendC::Reg::Abs(dstReg, srcReg, mask);
            AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

- [Atan2 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/atan2_950.md)
- [GeluQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/gelu_quant_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [DequantSwigluQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant_950.md)
- [AlltoAllQuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md)
- [DynamicQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_quant_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
- [QuantMax 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/quant_max_950.md)
- [MoeInitRoutingV3 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/moe/moe_init_routing_950.md)
- [DequantSwigluQuant 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/quant/dequant_swiglu_quant_950.md)
