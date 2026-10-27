---
type: API
title: ReleaseEventID
description: 用于释放HardEvent硬件同步事件的TEventID，通常与AllocEventID搭配使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/ReleaseEventID.md
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

# ReleaseEventID<a name="ZH-CN_TOPIC_0000001744571864"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

用于释放HardEvent（硬件类型同步事件）的TEventID，通常与[AllocEventID](alloc_event_id.md)搭配使用。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <HardEvent evt>
__aicore__ inline void ReleaseEventID(TEventID id)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

<a name="table58993346461"></a>
<table><thead align="left"><tr id="row88991534124617"><th class="cellrowborder" valign="top" width="12.23%" id="mcps1.2.3.1.1"><p id="p5899163484616"><a name="p5899163484616"></a><a name="p5899163484616"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.77000000000001%" id="mcps1.2.3.1.2"><p id="p1889953419469"><a name="p1889953419469"></a><a name="p1889953419469"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18991634144619"><td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.3.1.1 "><p id="p118996346463"><a name="p118996346463"></a><a name="p118996346463"></a>evt</p>
</td>
<td class="cellrowborder" valign="top" width="87.77000000000001%" headers="mcps1.2.3.1.2 "><p id="p1389943411468"><a name="p1389943411468"></a><a name="p1389943411468"></a>HardEvent硬件同步类型。该类型的具体说明请参考<a href="../../sync_control/intra_core_sync/set_flag_wait_flag_isasi.md">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

**表2**  参数说明

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p18393187172718"><a name="p18393187172718"></a><a name="p18393187172718"></a>id</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p1461163910269"><a name="p1461163910269"></a><a name="p1461163910269"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p143308311967"><a name="p143308311967"></a><a name="p143308311967"></a>TEventID类型，调用<a href="alloc_event_id.md">AllocEventID</a>申请获得的TEventID。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

AllocEventID、ReleaseEventID需成对出现，ReleaseEventID传入的TEventID需由对应的AllocEventID申请而来。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section882216131071"></a>

```cpp
AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>(); //需要插入scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventID);
......
......
......
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventID);
GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(eventID); //释放scalar等vector的同步HardEvent的ID
......
```

- [InplaceTril 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/inplace_tril_950.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [InplaceTriu 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/inplace_triu_950.md)
- [SwigluGroupQuantGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_grad_950.md)
- [FlashAttn 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attn_950.md)
- [Tril 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/tril_950.md)
- [DeformableConv2d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/deformable_conv2d.md)
- [IncreFlashAttention A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/incre_flash_attention.md)
- [Triu 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/triu_950.md)
