---
type: API
title: SetMaskCount
description: 设置Mask模式为Counter模式，直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.masking
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/mask_operations/SetMaskCount.md
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

# SetMaskCount<a name="ZH-CN_TOPIC_0000001790047006"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

设置Mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。本接口推荐配合API中isSetMask模板参数使用，当isSetMask为false时，支持用户调用本接口手动管理Counter模式，并通过[SetVectorMask](set_vector_mask.md)设置Counter模式下参与计算的元素个数。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetMaskCount()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 本接口推荐配合API中IsSetMask模板参数使用，当IsSetMask为false时，支持用户调用本接口手动管理Counter模式并通过[SetVectorMask](set_vector_mask.md)设置Counter模式下参与计算的元素个数。

- 设置为Counter模式的场景，需要在矢量计算使用完之后调用[SetMaskNorm](set_mask_norm.md)将Mask模式恢复为Normal模式，以避免影响后续计算。


- 针对Ascend 950PR&950DT系列产品，Mask为软仿实现，Tensor前n个数据计算接口内部不会将Mask模式切换为Normal模式。若当前场景已通过本接口设置为Counter模式，接口调用后仍为Counter模式；后续计算需要使用Normal模式时，需手动调用[SetMaskNorm](set_mask_norm.md)进行设置。


## 调用示例<a name="section837496171220"></a>

请参考[Counter模式调用示例](set_vector_mask.md#section837496171220)。

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [DequantSwigluQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_swiglu_quant.md)
- [FlashAttentionVarLenScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_var_len_score_950.md)
- [DynamicQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dynamic_quant.md)
- [grouped_matmul_swiglu_quant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant.md)
