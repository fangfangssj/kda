---
type: API
title: GetHcclContext
description: 获取指定Index通信域的context（消息区）地址。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL-Context/GetHcclContext.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
- 310p
---

# GetHcclContext

## 功能说明

获取指定Index通信域的context（消息区）地址。

## 函数原型

```
template <uint32_t index>
__aicore__ inline __gm__ uint8_t* __gm__ GetHcclContext(void)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| index | 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。 |

## 返回值说明

指定通信域的context（消息区）地址。

## 约束说明

当前最多只支持2个通信域。

## 调用示例

```
GM_ADDR contextGM = AscendC::GetHcclContext<0>();
```

- [AllGatherMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/all_gather_matmul.md)
- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul.md)
- [AlltoAllvGroupedMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_allv_grouped_mat_mul.md)
- [AttentionToFfn A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/attention_to_ffn.md)
- [BatchMatMulReduceScatterAlltoAll A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/batch_mat_mul_reduce_scatter_allto_all.md)
- [DistributeBarrier A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/distribute_barrier.md)
- [FfnToAttention A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/ffn_to_attention.md)
- [GroupedMatMulAllReduce A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/grouped_mat_mul_all_reduce.md)
