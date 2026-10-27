---
type: API
title: GetWindowsInAddr
description: 获取卡间通信数据WindowsIn起始地址，可用来直接作为计算的输入输出地址，减少拷贝。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/GetWindowsInAddr.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# GetWindowsInAddr

## 功能说明

获取卡间通信数据WindowsIn起始地址，可用来直接作为计算的输入输出地址，减少拷贝。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](../../../basic_api/tool_interface/system_resources_and_variables/get_block_idx.md)指定其在某一个核上运行。

## 函数原型

```
__aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| rankId | 输入 | 待查询的卡的Id。 |

## 返回值说明

返回对应卡的卡间通信数据WindowsIn起始地址。当rankId非法时，返回nullptr。

## 约束说明

无

## 调用示例

```
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
hccl.InitV2(contextGM, &tilingData);

auto winInAddr = hccl.GetWindowsInAddr(0);
auto winOutAddr = hccl.GetWindowsOutAddr(0);
auto rankId = hccl.GetRankId();
auto rankDim = hccl.GetRankDim(); // 4张卡
```

- [InplaceMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_matmul_all_reduce_add_rms_norm.md)
- [InplaceQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_quant_matmul_all_reduce_add_rms_norm.md)
- [InplaceWeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/inplace_weight_quant_matmul_all_reduce_add_rms_norm.md)
- [MatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_add_rms_norm.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
- [QuantMatmulAllReduce A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce.md)
- [QuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce_add_rms_norm.md)
- [WeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/weight_quant_matmul_all_reduce_add_rms_norm.md)
