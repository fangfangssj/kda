---
type: API
title: GetPhyAddr
description: 返回LocalTensor的地址或指定偏移量后的地址。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetPhyAddr.md
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

# GetPhyAddr<a name="ZH-CN_TOPIC_0000002330927386"></a>

## 功能说明<a name="section618mcpsimp"></a>

返回LocalTensor的地址或指定偏移量后的地址。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline uint64_t GetPhyAddr() const
__aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p195214291733"><a name="p195214291733"></a><a name="p195214291733"></a>偏移量。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor的地址或指定偏移量后的地址。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例
// 调用GetPhyAddr()返回LocalTensor地址，CPU上返回的是指针类型(T*)，NPU上返回的是物理存储的地址(uint64_t)
#ifdef ASCENDC_CPU_DEBUG
float *inputLocalCpuPtr = inputLocal.GetPhyAddr();
uint64_t realAddr = (uint64_t)inputLocalCpuPtr - (uint64_t)(GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(AscendC::TPosition::VECCALC)));
#else
uint64_t realAddr = inputLocal.GetPhyAddr();
#endif
```

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [GeGluBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/ge_glu_backward.md)
- [BlockSparseAttentionGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_grad_950.md)
- [DiagFlat A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/diag_flat.md)
- [GeGluBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/ge_glu_backward_950.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
