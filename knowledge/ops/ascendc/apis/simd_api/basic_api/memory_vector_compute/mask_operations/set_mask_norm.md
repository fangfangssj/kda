---
type: API
title: SetMaskNorm
description: 设置Mask模式为Normal模式（系统默认），支持开发者配置迭代次数和掩码。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.masking
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/mask_operations/SetMaskNorm.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_common.h
---

# SetMaskNorm<a name="ZH-CN_TOPIC_0000001836766205"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

设置Mask模式为Normal模式。该模式为系统默认模式，支持开发者配置迭代次数。本接口推荐配合API中isSetMask模板参数使用，当isSetMask为false时，支持用户调用本接口手动管理Normal模式，并通过[SetVectorMask](set_vector_mask.md)设置Normal模式下的掩码。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetMaskNorm()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 本接口推荐配合API中IsSetMask模板参数使用，当isSetMask为false时，支持用户调用本接口手动管理Normal模式并通过[SetVectorMask](set_vector_mask.md)设置Normal模式下的掩码。


- 针对Ascend 950PR&950DT系列产品，Mask为软仿实现，Tensor前n个数据计算接口内部不会将Mask模式切换为Normal模式。若当前场景已通过[SetMaskCount](set_mask_count.md)设置为Counter模式，接口调用后仍为Counter模式；后续计算需要使用Normal模式时，需手动调用本接口进行设置。


## 调用示例<a name="section837496171220"></a>

请参考[Normal模式调用示例](set_vector_mask.md#section837496171220)。

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
- [FlashAttentionScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_grad.md)
- [AddLayerNorm A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_layer_norm.md)
- [FlashAttentionScoreGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_grad_950.md)
- [FastLayerNorm A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/fast_layer_norm.md)
- [FlashAttentionUnpaddingScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_unpadding_score_grad.md)
- [DequantSwigluQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant.md)
