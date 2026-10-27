---
type: API
title: FetchEventID
description: 根据HardEvent获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/FetchEventID.md
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

# FetchEventID<a name="ZH-CN_TOPIC_0000001791531573"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <HardEvent evt>
__aicore__ inline TEventID FetchEventID()
__aicore__ inline TEventID FetchEventID(HardEvent evt)
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.1.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.1.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.1.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.1.4.1.1 "><p id="p143161011192917"><a name="p143161011192917"></a><a name="p143161011192917"></a>evt</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p9331531865"><a name="p9331531865"></a><a name="p9331531865"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.1.4.1.3 "><p id="p143308311967"><a name="p143308311967"></a><a name="p143308311967"></a>HardEvent类型，硬件同步类型。</p>
<p id="p15571184910124"><a name="p15571184910124"></a><a name="p15571184910124"></a>该类型的具体说明请参考<a href="../../sync_control/intra_core_sync/set_flag_wait_flag_isasi.md">SetFlag/WaitFlag(ISASI)</a>中同步类型的说明。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

相比于[AllocEventID](alloc_event_id.md)，FetchEventID适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。比如相同流水连续调用SetFlag/WaitFlag，如果两次传入的ID都是使用FetchEventID获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用AllocEventID。

## 返回值说明<a name="section640mcpsimp"></a>

TEventID

## 调用示例<a name="section6191129670"></a>

```cpp
AscendC::TEventID eventIdVToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S); //需要插scalar等vector的同步，申请对应的HardEvent的ID
AscendC::SetFlag<AscendC::HardEvent::V_S>(eventIdVToS);
AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventIdVToS);
```

- [DequantBias A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/dequant_bias.md)
- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [UpsampleBilinear2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa.md)
- [CircularPad2d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d.md)
- [GeGluBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/ge_glu_backward.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)

## 配套 API 与常见组合

- [GetTPipePtr](../get_t_pipe_ptr.md)——FetchEventID 是 TPipe 成员接口，kernel 内经 GetTPipePtr() 调用，证据行即 GetTPipePtr()->FetchEventID 同一行代码
- [SetFlag/WaitFlag](../../sync_control/intra_core_sync/set_flag_wait_flag_isasi.md)——FetchEventID 返回的 TEventID 直接作为 SetFlag/WaitFlag 入参，卡片示例即此链
