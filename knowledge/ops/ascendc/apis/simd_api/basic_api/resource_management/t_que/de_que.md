---
type: API
title: DeQue
description: 将Tensor从队列中取出用于后续处理，支持non-inplace和inplace两种接口，可通过TQueBind指定源和目的位置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQue/DeQue.md
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

# DeQue<a name="ZH-CN_TOPIC_0000001425890198"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

将Tensor从队列中取出，用于后续处理。

## 函数原型<a name="section620mcpsimp"></a>

-   无需指定源和目的位置
    -   non-inplace接口：将入队的LocalTensor地址从队列中取出赋值给新创建的Tensor并返回。

        ```cpp
        template <typename T>
        __aicore__ inline LocalTensor<T> DeQue()
        ```

    -   inplace接口：通过出参的方式返回，可以减少Tensor反复创建的开销，具体使用指导可参考[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)。

        ```cpp
        template <typename T>
        __aicore__ inline void DeQue(LocalTensor<T>& tensor)
        ```

-   需要指定源和目的位置

    通过[TQueBind](../t_que_bind/index.md)绑定VECIN和VECOUT可实现VECIN和VECOUT内存复用，如下接口用于存在Vector计算的场景下实现复用，在出队时需要指定源和目的位置；不存在Vector计算的场景下可直接调用LocalTensor<T\> DeQue\(\)出队接口。

    ```cpp
    template <TPosition srcUserPos, TPosition dstUserPos, typename T>
    __aicore__ inline LocalTensor<T> DeQue()
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
<tr id="row1629562101416"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p174916484213"><a name="p174916484213"></a><a name="p174916484213"></a>srcUserPos</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p24912414422"><a name="p24912414422"></a><a name="p24912414422"></a>用户指定队列的src position，当前只支持如下通路：GM-&gt;Unified Buffer（UB，VECIN/VECOUT）-&gt;GM。</p>
</td>
</tr>
<tr id="row1780112318146"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p1469019111420"><a name="p1469019111420"></a><a name="p1469019111420"></a>dstUserPos</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p96907110428"><a name="p96907110428"></a><a name="p96907110428"></a>用户指定队列的dst position，当前只支持如下通路：GM-&gt;UB（VECIN/VECOUT）-&gt;GM。</p>
</td>
</tr>
</tbody>
</table>

**表2**  参数说明

<a name="table181221135162517"></a>
<table><thead align="left"><tr id="row151221135112520"><th class="cellrowborder" valign="top" width="12.471247124712471%" id="mcps1.2.4.1.1"><p id="p1353754532512"><a name="p1353754532512"></a><a name="p1353754532512"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p1253774516259"><a name="p1253774516259"></a><a name="p1253774516259"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.93749374937492%" id="mcps1.2.4.1.3"><p id="p1653710452259"><a name="p1653710452259"></a><a name="p1653710452259"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12122235102511"><td class="cellrowborder" valign="top" width="12.471247124712471%" headers="mcps1.2.4.1.1 "><p id="p1537164502512"><a name="p1537164502512"></a><a name="p1537164502512"></a>tensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1653714592515"><a name="p1653714592515"></a><a name="p1653714592515"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p159171759152612"><a name="p159171759152612"></a><a name="p159171759152612"></a>inplace接口需要通过出参的方式返回Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   对空队列执行DeQue是一种异常行为，会在CPU调测时报错。
-   non-inplace接口和指定源和目的位置的接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

## 返回值说明<a name="section640mcpsimp"></a>

non-inplace接口和指定源和目的位置的接口返回值为从队列中取出的[LocalTensor](../../data_structures/local_tensor/index.md)；inplace接口没有返回值。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
// 接口: DeQue Tensor
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 4> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue(tensor1);
AscendC::LocalTensor<half> tensor2 = que.DeQue<half>(); // 将tensor从VECOUT的Queue中搬出
// 接口: DeQue Tensor，指定特定的Src/Dst position
AscendC::TPipe pipe;
AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> que;
int num = 4;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>();
que.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>(tensor1);
// 将tensor从VECIN的Queue中搬出
AscendC::LocalTensor<half> tensor2 = que.DeQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN, half>(); 
// inplace接口
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 0> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len);
AscendC::LocalTensor<half> tensor1;
que.AllocTensor<half>(tensor1);
que.EnQue(tensor1);
que.DeQue<half>(tensor1); // 将tensor从VECOUT的Queue中搬出
que.FreeTensor<half>(tensor1);
```

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [Celu 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/celu_950.md)
- [AttentionUpdate A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/attention_update.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [DiagFlat A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/diag_flat.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [UpsampleBicubic2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa.md)
- [InplaceTril 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/inplace_tril_950.md)

## 配套 API 与常见组合

- [TQue简介](t_que_intro.md)
- [AllocTensor](alloc_tensor.md)——同一队列上 tensor 的生产侧 AllocTensor 与消费侧 DeQue，构成 Alloc-EnQue-DeQue-Free 四步链的分配与取出
- [EnQue](en_que.md)——官方文档明文 EnQue/DeQue 必须成对匹配使用：队列流水通信的生产/消费两端自动确保同步
- [FreeTensor](free_tensor.md)——消费侧相邻步骤 DeQue-使用-FreeTensor 归还
- [InitBuffer](../t_pipe/init_buffer.md)——TPipe::InitBuffer 在 TPipe 上为队列分配内存（pipe.InitBuffer）
- [TQueBind 复用中的内存收益与中间计算](../../../../../guides/memory/tquebind_reuse_and_intermediate_compute.md)
- [双缓冲物理槽与常驻参数预算](../../../../../guides/memory/double_buffer_slot_counts_and_resident_parameters.md)
- [InitBuffer](../t_buf_pool/init_buffer.md)——TBufPool::InitBuffer 分配 DeQue 依赖的队列 Buffer 及其中同步事件；调用对象是资源池，不是 TPipe
- [TPipe简介](../t_pipe/t_pipe_intro.md)——DeQue 依赖的队列内存与同步由 kernel 唯一 TPipe 统一管理（P(a|b)=1.00）
