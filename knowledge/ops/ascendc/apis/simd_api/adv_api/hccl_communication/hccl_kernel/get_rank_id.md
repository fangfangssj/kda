---
type: API
title: GetRankId
description: 获取本卡RankId。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/GetRankId.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# GetRankId

## 功能说明

获取本卡RankId。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](../../../basic_api/tool_interface/system_resources_and_variables/get_block_idx.md)指定其在某一个核上运行。

## 函数原型

```
__aicore__ inline uint32_t GetRankId()
```

## 参数说明

无

## 返回值说明

返回本卡RankId。

## 约束说明

无

## 调用示例

请参见[GetWindowsInAddr](get_windows_in_addr.md)的调用示例。

- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
- [BatchMatMulReduceScatterAlltoAll A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/batch_mat_mul_reduce_scatter_allto_all.md)
- [InplaceMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_matmul_all_reduce_add_rms_norm.md)
- [InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_quant_matmul_all_reduce_add_rms_norm.md)
- [InplaceWeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_weight_quant_matmul_all_reduce_add_rms_norm.md)
- [MatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_add_rms_norm.md)
- [QuantMatmulAllReduce A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce.md)
- [QuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce_add_rms_norm.md)
- [WeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/weight_quant_matmul_all_reduce_add_rms_norm.md)
