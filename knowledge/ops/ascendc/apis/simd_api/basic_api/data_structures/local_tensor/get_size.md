---
type: API
title: GetSize
description: 获取当前LocalTensor Size大小。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
---

# GetSize<a name="ZH-CN_TOPIC_0000002330927374"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取当前LocalTensor Size大小。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline uint32_t GetSize() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

当前LocalTensor Size大小。单位为元素。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

参考[调用示例](local_tensor_constructor.md#section17531157161314)。

```cpp
// 示例
auto size = inputLocal.GetSize(); // 获取inputLocal的长度，size大小为inputLocal有多少个元素
// 示例5结果如下：
// size大小为srcLen，256。
```

- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [DiagFlat A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/diag_flat.md)
- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
- [IncreFlashAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/incre_flash_attention.md)
- [UpsampleBilinear2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa.md)
- [WeightQuantBatchMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul_950.md)
- [InplaceWeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_weight_quant_matmul_all_reduce_add_rms_norm.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [WeightQuantBatchMatmulNz 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul_nz_950.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
