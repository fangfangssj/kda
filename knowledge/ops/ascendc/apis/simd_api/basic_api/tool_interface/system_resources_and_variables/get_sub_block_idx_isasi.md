---
type: API
title: GetSubBlockIdx
description: 在分离模式架构下，获取逻辑AI Core上AIC或AIV的逻辑索引。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSubBlockIdx_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetSubBlockIdx(ISASI)

## 功能说明

头文件路径为： `"basic_api/kernel_operator_sys_var_intf.h"`。

在[分离模式架构](../../../../../guides/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)下，获取逻辑AI Core上Cube Core（AIC）或者Vector Core（AIV）的逻辑索引。

GetSubBlockIdx在使用`__mix__(1, 2)`作为[函数执行空间限定符](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#section1074418132518)的场景下，可以用于区分同一个核上的两个不同Vector Core，取值分别为0和1。

## 函数原型

```cpp
__aicore__ inline int64_t GetSubBlockIdx()
```

## 参数说明

无

## 返回值说明

返回逻辑AI Core上Cube Core（AIC）或者Vector Core（AIV）的逻辑索引。

SubBlockIdx的取值范围为\[0, SubBlockNum\)，SubBlockNum可以通过[GetSubBlockNum](get_sub_block_num_isasi.md)获取。

## 约束说明

无

## 调用示例

```cpp
int64_t subBlockID = AscendC::GetSubBlockIdx();
```

- [QuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/quant_matmul_950.md)
- [BlockSparseAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention.md)
- [QuantMatmulWeightNz 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/quant_matmul_weight_nz_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [RotateQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/rotate_quant_950.md)
- [GroupedMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_950.md)
- [TransposeQuantBatchMatMul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_quant_batch_mat_mul_950.md)
- [GroupedMatmulFinalizeRouting 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_finalize_routing_950.md)
- [TransposeQuantBatchMatMulWeightNz 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_quant_batch_mat_mul_weight_nz_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
