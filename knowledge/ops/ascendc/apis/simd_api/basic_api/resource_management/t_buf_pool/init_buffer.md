---
type: API
title: InitBuffer
description: 为TQue或TBuf进行内存分配，支持指定内存块个数和大小。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TBufPool/InitBuffer.md
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

# InitBuffer<a name="ZH-CN_TOPIC_0000001866643004"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

调用TBufPool::InitBuffer接口为TQue/TBuf进行内存分配。

## 函数原型<a name="section620mcpsimp"></a>

-   为TQue分配内存

    ```cpp
    template <class T>
    __aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
    ```

-   为TBuf分配内存

    ```cpp
    template <TPosition pos>
    __aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  为TQue分配内存的函数原型-模板参数说明

| 参数名称 | 含义 |
| -------- | ---- |
| T | que参数的类型。|

**表2**  为TQue分配内存的函数原型-接口参数说明

<a name="table193329316393"></a>
<table><thead align="left"><tr id="row123331131153919"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p8333133153913"><a name="p8333133153913"></a><a name="p8333133153913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p518118718459"><a name="p518118718459"></a><a name="p518118718459"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p833353113393"><a name="p833353113393"></a><a name="p833353113393"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11660173845017"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p466053810507"><a name="p466053810507"></a><a name="p466053810507"></a>que</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p885774605014"><a name="p885774605014"></a><a name="p885774605014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p0660153818501"><a name="p0660153818501"></a><a name="p0660153818501"></a>需要分配内存的TQue对象</p>
</td>
</tr>
<tr id="row03336319398"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p11399116193313"><a name="p11399116193313"></a><a name="p11399116193313"></a>num</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p111819774511"><a name="p111819774511"></a><a name="p111819774511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p6383173514333"><a name="p6383173514333"></a><a name="p6383173514333"></a>分配内存块的个数</p>
</td>
</tr>
<tr id="row1430772593316"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p530752514330"><a name="p530752514330"></a><a name="p530752514330"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p530711252335"><a name="p530711252335"></a><a name="p530711252335"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p183081251336"><a name="p183081251336"></a><a name="p183081251336"></a>每个内存块的大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐</p>
</td>
</tr>
</tbody>
</table>

**表3**  为TBuf分配内存的函数原型-模板参数说明

| 参数名称 | 含义 |
| -------- | ---- |
| pos | Buffer逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。关于TPosition的具体介绍请参考[TPosition](../../aux_data_structures/t_position.md)。|

**表4**  为TBuf分配内存的函数原型-接口参数说明

<a name="table5376122715308"></a>
<table><thead align="left"><tr id="row1337716275309"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p1537762711305"><a name="p1537762711305"></a><a name="p1537762711305"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.2.4.1.2"><p id="p153771127123013"><a name="p153771127123013"></a><a name="p153771127123013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.27000000000001%" id="mcps1.2.4.1.3"><p id="p17377162715303"><a name="p17377162715303"></a><a name="p17377162715303"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row19377627133012"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p737710279307"><a name="p737710279307"></a><a name="p737710279307"></a>buf</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p13377122733010"><a name="p13377122733010"></a><a name="p13377122733010"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p19377102793016"><a name="p19377102793016"></a><a name="p19377102793016"></a>需要分配内存的TBuf对象</p>
</td>
</tr>
<tr id="row13377162793019"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p5377527113018"><a name="p5377527113018"></a><a name="p5377527113018"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p12377122712304"><a name="p12377122712304"></a><a name="p12377122712304"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p6514716314"><a name="p6514716314"></a><a name="p6514716314"></a>为TBuf分配的内存大小，单位为Bytes，非32Bytes对齐会自动向上补齐至32Bytes对齐</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

声明TBufPool时，可以通过bufIDSize指定可分配Buffer的最大数量，默认上限为4，最大为16。TQue或TBuf的物理内存需要和TBufPool一致。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

参考[InitBufPool](init_buf_pool.md)。

- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [MoeDistributeDispatch 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_950.md)
- [NormRopeConcatGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/norm_rope_concat_grad.md)
- [NormRopeConcatGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/norm_rope_concat_grad_950.md)

## 配套 API 与常见组合

- [InitBuffer](../t_pipe/init_buffer.md)——同名 TPipe::InitBuffer 在 TPipe 上调用（pipe.InitBuffer）。本卡是资源池接口，示例为 tbufPool.InitBuffer，且 TQue/TBuf 物理内存须与该池一致
- [AllocTensor](../t_que/alloc_tensor.md)——AllocTensor 分配的 tensor 大小即 InitBuffer 设定的每块内存长度（AllocTensor 卡明文），未 InitBuffer 的队列无内存可分配
- [FreeTensor](../t_que/free_tensor.md)——FreeTensor 归还的 tensor 回到 InitBuffer 为该队列分配的内存池，FreeTensor 卡配套节已列 InitBuffer
- [TQue简介](../t_que/t_que_intro.md)——本卡 TBufPool::InitBuffer 的 que 参数即 TQue，调用对象是资源池
- [DeQue](../t_que/de_que.md)——DeQue 收发所依赖的队列 Buffer 及内含同步事件由 InitBuffer 分配（TQue 卡明文 Buffer 存储 eventID）
- [EnQue](../t_que/en_que.md)——EnQue 入队依赖 InitBuffer 分配的队列 Buffer 及其同步事件
