---
type: API
title: AICore（OpDef）
description: 设置AICore的配置信息：包括Tiling处理回调函数、能力检查回调函数、基础的配置信息等。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.implementation.operator_registration
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/prototype_register_management/OpDef/AICore.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# AICore<a name="ZH-CN_TOPIC_0000002114097049"></a>

## 功能说明<a name="zh-cn_topic_0000001626208081_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section36583473819"></a>

设置AICore的配置信息：包括Tiling处理回调函数、能力检查回调函数、基础的配置信息等。

## 函数原型<a name="zh-cn_topic_0000001626208081_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreDef &AICore(void)
```

## 参数说明<a name="zh-cn_topic_0000001626208081_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001626208081_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreDef请参考[OpAICoreDef](../op_ai_core_def/index.md)。

## 约束说明<a name="zh-cn_topic_0000001626208081_zh-cn_topic_0000001526594958_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

- [UpsampleLinear1dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_linear_1d_backward_950.md)
- [Cat 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/cat_950.md)
- [InplaceSigmoid A2 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/inplace_sigmoid.md)
- [FlashAttentionVarLenScore 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_var_len_score_950.md)
- [ClampMin 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/clamp_min_950.md)
- [Swish A2 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/swish.md)
- [NsaCompress A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/nsa_compress.md)
- [InplaceTril 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/inplace_tril_950.md)
- [SwishBackward 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/swish_backward_950.md)
- [MhcPreBackward 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mhc/mhc_pre_backward_950.md)
