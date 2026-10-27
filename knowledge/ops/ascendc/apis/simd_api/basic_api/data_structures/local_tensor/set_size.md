---
type: API
title: SetSize
description: 设置当前LocalTensor Size大小。单位为元素。当用户重用local tensor变量且使用长度发生变化的时候，需要使用此接口重新设置Size。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/SetSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tensor.h
---

# SetSize<a name="ZH-CN_TOPIC_0000002364845729"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置当前LocalTensor Size大小。单位为元素。当用户重用local tensor变量且使用长度发生变化的时候，需要使用此接口重新设置Size。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetSize(const uint32_t size)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.98%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.08%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p2378846125714"><a name="p2378846125714"></a><a name="p2378846125714"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p358962315587"><a name="p358962315587"></a><a name="p358962315587"></a>元素个数，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例1 SetSize如下示例将申请的Tensor长度修改为256(单位为元素)
AscendC::LocalTensor<float> tmpBuffer3 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer3.SetSize(256);
```

- [GroupedBiasAddGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/math/grouped_bias_add_grad.md)
- [WeightQuantBatchMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul.md)
- [IncreFlashAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/incre_flash_attention.md)
- [MaskedSoftmaxWithRelPosBias A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/masked_softmax_with_rel_pos_bias.md)
- [NsaCompress A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/nsa_compress.md)
- [ModulateBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/vfusion/modulate_backward.md)
- [MoeTokenPermuteWithRoutingMap 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/moe/moe_token_permute_with_routing_map_950.md)
- [ScaledMaskedSoftmaxBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/vfusion/scaled_masked_softmax_backward.md)
