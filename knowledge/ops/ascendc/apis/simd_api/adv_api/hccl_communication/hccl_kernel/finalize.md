---
type: API
title: Finalize（HCCL集合通信）
description: 通知服务端后续无通信任务，执行结束后退出。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/Finalize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
---

# Finalize

## 功能说明

通知服务端后续无通信任务，执行结束后退出。

## 函数原型

```
template <bool sync = true>
__aicore__ inline void Finalize()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sync | 输入 | 是否需要等待服务端的通信完成。bool类型，参数取值如下：<br>true：默认值，表示客户端将检测并等待最后一个通信任务完成。<br>false：表示客户端发送Finalize通知后，不等待服务端通信任务完成即退出。该取值仅跳过客户端侧的阻塞等待，不会取消已被服务端接收并编排的通信任务；但不表示已下发的通信任务已完成，也不保证接口调用结束时对端已收到数据。 |

## 返回值说明

无

## 约束说明

-   调用本接口前确保已调用过[InitV2](init_v2.md)和[SetCcTilingV2](set_cc_tiling_v2.md)接口。
-   本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。
-   调用Finalize\<false\>()前，若需要保证通信任务完成，必须先完成同步；否则客户端直接退出后，无法保证数据传输完成。多队列[BatchWrite](batch_write.md)场景需先对所有队列调用[QueueBarrier](queue_barrier.md)接口，等待队列上的BatchWrite任务完成后再调用Finalize\<false\>()，请参见[BatchWrite的多个队列的点对点通信调用示例](batch_write.md#batchwrite_multi_queue)。
-   Finalize为终止接口，调用后服务端开始退出，当前Hccl对象不可再次用于通信。如需继续通信，需重新创建Hccl对象并调用[InitV2](init_v2.md)接口初始化。

## 调用示例

```
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
hccl.InitV2(contextGM, &tilingData);
hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, mc2CcTiling));
if (AscendC::g_coreType == AIC) {
    HcclHandle handleId =
        hccl.ReduceScatter(sendBuf, recvBuf, 100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 10);
    hccl.Commit(handleId); // 通知服务端可以执行上述的ReduceScatter任务
    for (uint8_t i = 0; i < 10; i++) {
        hccl.Wait(handleId); // 阻塞接口，需等待上述ReduceScatter任务执行完毕
    }
    hccl.Finalize<true>(); // 后续无其他通信任务，通知服务端执行上述ReduceScatter任务之后即可以退出
}
```

- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md)
- [BatchMatMulReduceScatterAlltoAll A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/batch_mat_mul_reduce_scatter_allto_all.md)
- [InplaceMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_matmul_all_reduce_add_rms_norm.md)
- [InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_quant_matmul_all_reduce_add_rms_norm.md)
- [InplaceWeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_weight_quant_matmul_all_reduce_add_rms_norm.md)
- [MatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_add_rms_norm.md)
- [MatmulAlltoAll A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all.md)
- [MatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all_950.md)
