---
type: API
title: AllocTensor
description: 从Que中分配Tensor，Tensor大小为InitBuffer时设置的每块内存长度，支持non-inplace和inplace两种接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TQue/AllocTensor.md
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

# AllocTensor<a name="ZH-CN_TOPIC_0000001475776125"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

从Que中分配Tensor，Tensor所占大小为InitBuffer时设置的每块内存长度。

## 函数原型<a name="section620mcpsimp"></a>

-   non-inplace接口：构造新的Tensor作为内存管理的对象

    ```cpp
    template <typename T>
    __aicore__ inline LocalTensor<T> AllocTensor()
    ```

-   inplace接口：直接使用传入的Tensor作为内存管理的对象，可以减少Tensor反复创建的开销，具体使用指导可参考[Tensor原地操作](../../../../../guides/programming_guide/appendix/common_operations/tensor_inplace_performance.md)。

    ```cpp
    template <typename T>
    __aicore__ inline void AllocTensor(LocalTensor<T>& tensor)
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
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1653714592515"><a name="p1653714592515"></a><a name="p1653714592515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.93749374937492%" headers="mcps1.2.4.1.3 "><p id="p165371945142512"><a name="p165371945142512"></a><a name="p165371945142512"></a>inplace接口需要传入<a href="../../data_structures/local_tensor/index.md">LocalTensor</a>作为内存管理的对象。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   同一个TPosition上的所有Queue，连续调用AllocTensor接口申请的Tensor数量，根据AI处理器型号的不同，有数量约束。申请Buffer时，需要满足该约束。
    
    - Ascend 950PR&950DT系列产品不超过8个。
    
    
    - Atlas A3系列产品不超过8个。
    
    
    - Atlas A2系列产品不超过8个。
    
    
    - Atlas 200I/500 A2推理产品不超过8个。
    
    
    - Atlas推理系列产品AI Core不超过8个。
    
    
    - Atlas推理系列产品Vector Core不超过8个。
    
    
    - Atlas训练系列产品不超过4个。
    
    
    - Kirin X90处理器系列产品不超过8个。
    
    
    - Kirin 9030处理器系列产品不超过8个。
        
-   non-inplace接口分配的Tensor内容可能包含随机值。
-   non-inplace接口，需要将TQueBind的depth模板参数设置为非零值；inplace接口，需要将TQueBind的depth模板参数设置为0。

## 返回值说明<a name="section640mcpsimp"></a>

non-inplace接口返回值为LocalTensor对象，inplace接口没有返回值。

## 调用示例<a name="section642mcpsimp"></a>

-   示例一

    ```cpp
    // 使用AllocTensor分配Tensor
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
    int num = 2;
    int len = 1024;
    pipe.InitBuffer(que, num, len); // InitBuffer分配内存块数为2，每块大小为1024Bytes
    AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>(); // AllocTensor分配Tensor长度为1024Bytes
    ```

-   示例二

    ```cpp
    // 连续使用AllocTensor的限制场景举例如下:
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que3;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que4;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> que5;
    // 不建议：
    // 比如，算子有6个输入，需要申请6块buffer
    // 通过6个队列为其申请内存，分别为que0~que5，每个que分配1块,申请VECIN TPosition上的buffer总数为6
    // 假设，同一个TPosition上连续Alloc的Buffer数量限制为4，超出该限制后，使用AllocTensor/FreeTensor会出现分配资源失败
    // 在NPU上可能体现为卡死等异常行为，在CPU Debug场景会出现报错提示
    pipe.InitBuffer(que0, 1, len);
    pipe.InitBuffer(que1, 1, len);
    pipe.InitBuffer(que2, 1, len);
    pipe.InitBuffer(que3, 1, len);
    pipe.InitBuffer(que4, 1, len);
    pipe.InitBuffer(que5, 1, len);
    
    AscendC::LocalTensor<T> local1 = que0.AllocTensor<T>();
    AscendC::LocalTensor<T> local2 = que1.AllocTensor<T>();
    AscendC::LocalTensor<T> local3 = que2.AllocTensor<T>();
    AscendC::LocalTensor<T> local4 = que3.AllocTensor<T>();
    // 第5个AllocTensor会出现资源分配失败，同一个TPosition上同时Alloc出来的Tensor数量超出了4个的限制
    AscendC::LocalTensor<T> local5 = que4.AllocTensor<T>();
    
    // 此时建议通过以下方法解决：
    // 如果确实有多块buffer使用,可以将多个buffer合并到一块buffer,通过偏移使用
    pipe.InitBuffer(que0, 1, len * 3);
    pipe.InitBuffer(que1, 1, len * 3);
    /*
     * 分配出3块内存大小的LocalTensor, local1的地址为que0中buffer的起始地址，
     * local2的地址为local1的地址偏移len后的地址，local3的地址为local1的地址偏移
     * len * 2的地址
     */
    int32_t offset1 = len;
    int32_t offset2 = len * 2;
    AscendC::LocalTensor<T> local1 = que0.AllocTensor<T>();
    AscendC::LocalTensor<T> local2 = local1[offset1];
    AscendC::LocalTensor<T> local3 = local1[offset2];
    ```

-   示例三：inplace接口

    ```cpp
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> que;
    int num = 2;
    int len = 1024;
    pipe.InitBuffer(que, num, len); // InitBuffer分配内存块数为2，每块大小为1024Bytes
    AscendC::LocalTensor<half> tensor1;
    que.AllocTensor<half>(tensor1); // AllocTensor分配Tensor长度为1024Bytes
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

- [TQue简介](t_que_intro.md)——AllocTensor 是 TQue 的成员接口，从 TQue 分配 tensor
- [InitBuffer](../t_pipe/init_buffer.md)——TPipe::InitBuffer 在 TPipe 上为队列分配内存（pipe.InitBuffer）
- [EnQue](en_que.md)——生产侧相邻步骤 AllocTensor-填充-EnQue，为官方 TQue 示例的 canonical 顺序
- [DeQue](de_que.md)——同一队列上 tensor 的生产侧 AllocTensor 与消费侧 DeQue，构成 Alloc-EnQue-DeQue-Free 四步链的分配与取出
- [FreeTensor](free_tensor.md)——官方文档明文 AllocTensor/FreeTensor 在切分资源池时必须成对匹配使用
- [Ascend C 缓冲所有权、流水交接与槽位布局](../../../../../guides/memory/buffer_ownership_and_slot_layout.md)
- [双缓冲物理槽与常驻参数预算](../../../../../guides/memory/double_buffer_slot_counts_and_resident_parameters.md)
- [InitBuffer](../t_buf_pool/init_buffer.md)——TBufPool::InitBuffer 设定每块内存长度，AllocTensor 按该长度分配；这不是 TPipe::InitBuffer
- [TPipe简介](../t_pipe/t_pipe_intro.md)——AllocTensor 的内存源是 kernel 唯一 TPipe 经 InitBuffer 绑定的队列内存（P(a|b)=1.00），缺 TPipe 则队列范式不成立
