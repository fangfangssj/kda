---
type: API
title: And
description: 根据mask对两个RegTensor或MaskReg执行按位与操作，将结果写入目的操作数，支持ZEROING模式。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/logical_compute/And.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h
- include/basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h
---

# And<a name="ZH-CN_TOPIC_0000001956986921"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"`（对`RegTensor`操作）、`"basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h"`（对`MaskReg`操作）。

本节介绍两种接口，分别用于对[RegTensor](../register_data_types/reg_tensor.md)和[MaskReg](../register_data_types/mask_reg.md)进行操作：

- 对`RegTensor`操作：

  根据mask对输入数据srcReg0、srcReg1执行按位与（&）操作，将结果写入dstReg。

- 对`MaskReg`操作：

  根据mask对两个输入`MaskReg`数据src0、src1执行按位与（&）操作，将结果写入dst。

## 函数原型<a name="section620mcpsimp"></a>

- 对`RegTensor`操作

  ```cpp
  template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
  __simd_callee__ inline void And(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
  ```

- 对`MaskReg`操作

  ```cpp
  __simd_callee__ inline void And(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
  ```

## 参数说明<a name="section622mcpsimp"></a>

- 对`RegTensor`操作

  **表1**  模板参数说明

  | 参数名 | 描述 |
  | :----- | :--- |
  | T | 操作数数据类型。 |
  | mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型。选择MERGING模式或ZEROING模式。<br>• ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>• MERGING模式当前不支持。 |
  | U | 目的操作数和源操作数的`RegTensor`类型，例如`RegTensor<uint32_t>`，由编译器自动推导，用户不需要填写。 |

  **表2**  参数说明

  | 参数名 | 输入/输出 | 描述 |
  | :----- | :-------- | :--- |
  | dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
  | srcReg0 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>两个源操作数的数据类型需要与目的操作数保持一致。 |
  | srcReg1 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>两个源操作数的数据类型需要与目的操作数保持一致。 |
  | mask | 输入 | 指示源操作数元素操作的有效性，详细说明请参考[MaskReg文档](../register_data_types/mask_reg.md)。 |

- 对MaskReg操作

  **表3**  参数说明

  | 参数名 | 描述 |
  | :----- | :--- |
  | dst | 目的操作数。 |
  | src0 | 源操作数。 |
  | src1 | 源操作数。 |
  | mask | 指示在计算过程中哪些bit有效。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 数据类型

支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

- 对`RegTensor`操作

  ```cpp
  template <typename T>
  __simd_vf__ inline void AndVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
  {
      AscendC::Reg::RegTensor<T> srcReg0;
      AscendC::Reg::RegTensor<T> srcReg1;
      AscendC::Reg::RegTensor<T> dstReg;
      AscendC::Reg::MaskReg mask;
      for (uint16_t i = 0; i < repeatTimes; i++) {
          mask = AscendC::Reg::UpdateMask<T>(count);
          AscendC::Reg::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
          AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
          AscendC::Reg::And(dstReg, srcReg0, srcReg1, mask);
          AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
      }
  }
  ```

- 对`MaskReg`操作

  ```cpp
  template <typename T>
  __simd_vf__ inline void AndVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
  {
      AscendC::Reg::RegTensor<T> srcReg;
      AscendC::Reg::MaskReg src0 = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALLF>();
      AscendC::Reg::MaskReg src1 = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
      AscendC::Reg::MaskReg dst;
      AscendC::Reg::MaskReg mask;
      for (uint16_t i = 0; i < repeatTimes; ++i) {
          mask = AscendC::Reg::UpdateMask<T>(count);
          AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
          AscendC::Reg::And(dst, src0, src1, mask);
          AscendC::Reg::Adds(srcReg, srcReg, 0, dst);
          AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
      }
  }
  ```

- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [QuantLightningIndexer 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/quant_lightning_indexer_950.md)
- [InplaceClampMax 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/inplace_clamp_max_950.md)
- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [All 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/all_950.md)
- [RmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_dynamic_mx_quant_950.md)
- [MoeDistributeDispatchSetup 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_setup_950.md)
- [Atan2 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/atan2_950.md)
