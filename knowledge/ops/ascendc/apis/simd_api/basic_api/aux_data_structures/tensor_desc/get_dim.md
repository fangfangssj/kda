---
type: API
title: GetDim
description: 获取Tensor的维度。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/aux_data_structures/TensorDesc/GetDim.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_list_tensor_intf.h
---

# GetDim<a name="ZH-CN_TOPIC_0000002340337961"></a>

## 功能说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section36583473819"></a>

获取Tensor的维度。

## 函数原型<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section13230182415108"></a>

```cpp
uint64_t GetDim()
```

## 参数说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section25791320141317"></a>

返回Tensor的维度。

## 约束说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section320753512363"></a>

参考[调用示例](get_data_ptr.md#zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section320753512363)。

- [ChunkCat A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/chunk_cat.md)
- [FusedSgd A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/optim/fused_sgd.md)
