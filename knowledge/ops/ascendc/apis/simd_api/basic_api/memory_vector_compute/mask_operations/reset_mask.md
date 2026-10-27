---
type: API
title: ResetMask
description: 恢复mask值为默认值（全1），表示矢量计算中每次迭代内所有元素都参与运算。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.masking
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/mask_operations/ResetMask.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_common.h
---

# ResetMask<a name="ZH-CN_TOPIC_0000001857638449"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为:`"basic_api/kernel_common.h"`。

恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void ResetMask()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

调用SetVectorMask接口之后，如需恢复mask的值为默认值（全1），可调用本接口实现。

## 调用示例<a name="section837496171220"></a>

用[SetVectorMask](set_vector_mask.md)设置mask值并使用完成后，使用ResetMask恢复mask的值为默认值。

```cpp
AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>(128);
AscendC::ResetMask();
```

更多示例请参考[接口外设置Mask](../simd_compute/mask.md#接口外设置mask)。

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
- [FlashAttentionVarLenScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_var_len_score_950.md)
- [AddLayerNorm A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_layer_norm.md)
- [grouped_matmul_swiglu_quant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant.md)
- [DequantSwigluQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant.md)
- [MoeTokenPermute A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/moe/moe_token_permute.md)
- [DynamicQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_quant.md)
