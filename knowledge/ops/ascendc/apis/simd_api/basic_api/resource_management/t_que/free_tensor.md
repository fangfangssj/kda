---
type: API
title: FreeTensor
description: 释放Que中的指定Tensor，需与AllocTensor配对使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQue/FreeTensor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_tpipe.h
---

# FreeTensor<a name="ZH-CN_TOPIC_0000001475769485"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

释放Que中的指定Tensor。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void FreeTensor(LocalTensor<T>& tensor)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="12.139999999999999%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.86%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p12101541625"><a name="p12101541625"></a><a name="p12101541625"></a><span>Tensor的数据类型，支持的类型请见<a href="../../data_structures/local_tensor/index.md">LocalTensor</a>相关描述。</span></p>
</td>
</tr>
</tbody>
</table>

**表2**  参数说明

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p466053810507"><a name="p466053810507"></a><a name="p466053810507"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p885774605014"><a name="p885774605014"></a><a name="p885774605014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p0660153818501"><a name="p0660153818501"></a><a name="p0660153818501"></a>待释放的Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 使用FreeTensor释放通过AllocTensor分配的Tensor，注意配对使用
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.FreeTensor<half>(tensor1);
```

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [Celu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/celu_950.md)
- [AttentionUpdate A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/attention_update.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [ClampMin 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/clamp_min_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [UpsampleBicubic2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa.md)
- [DiagFlat A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/diag_flat.md)

## 配套 API 与常见组合

- [TQue简介](t_que_intro.md)——FreeTensor 是 TQue 的成员接口，向 TQue 归还 tensor
- [AllocTensor](alloc_tensor.md)——官方文档明文 AllocTensor/FreeTensor 在切分资源池时必须成对匹配使用
- [EnQue](en_que.md)
- [DeQue](de_que.md)——消费侧相邻步骤 DeQue-使用-FreeTensor 归还
- [InitBuffer](../t_pipe/init_buffer.md)——TPipe::InitBuffer 在 TPipe 上为队列分配内存（pipe.InitBuffer）
- [Ascend C 缓冲所有权、流水交接与槽位布局](../../../../../guides/memory/buffer_ownership_and_slot_layout.md)
- [InitBuffer](../t_buf_pool/init_buffer.md)——TBufPool::InitBuffer 分配的内存池接收 FreeTensor 归还的 tensor，调用对象是资源池
- [TPipe简介](../t_pipe/t_pipe_intro.md)——FreeTensor 作用的队列内存池由 kernel 唯一 TPipe 统一管理，同一内存生命周期链（P(a|b)=1.00）
