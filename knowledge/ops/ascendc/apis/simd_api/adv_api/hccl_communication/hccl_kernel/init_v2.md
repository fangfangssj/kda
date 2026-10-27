---
type: API
title: InitV2
description: HCCL客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/InitV2.md
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

# InitV2

## 功能说明

HCCL客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](../../../basic_api/tool_interface/system_resources_and_variables/get_block_idx.md)指定其在某一个核上运行。

## 函数原型

```
__aicore__ inline void InitV2(GM_ADDR context, const void* initTiling)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| context | 输入 | 通信上下文，包含rankDim，rankID等相关信息。通过框架提供的获取通信上下文的接口[GetHcclContext](../hccl_context/get_hccl_context.md)获取context。 |
| initTiling | 输入 | 通信域初始化[Mc2InitTiling](../hccl_tiling/tiling_data_struct.md#table4835205712588)的地址。[Mc2InitTiling](../hccl_tiling/tiling_data_struct.md#table4835205712588)在Host侧计算得出，具体请参考[表1 Mc2InitTiling参数说明](../hccl_tiling/tiling_data_struct.md#table4835205712588)，由框架传递到核函数（Kernel）中使用。 |

## 返回值说明

无

## 约束说明

-   本接口必须与[SetCcTilingV2](set_cc_tiling_v2.md)接口配合使用。
-   调用本接口时，必须使用标准C++语法定义TilingData结构体的开发方式。
-   调用本接口传入的initTiling参数，不能使用Global Memory地址，建议通过[GET\_TILING\_DATA\_WITH\_STRUCT](../../../basic_api/kernel_tiling/get_tiling_data_with_struct.md)接口获取TilingData的栈地址。
-   本接口不支持使用相同的context初始化多个HCCL对象。

## 调用示例

用户自定义TilingData结构体：

```
class UserCustomTilingData {
    AscendC::tiling::Mc2InitTiling initTiling;
    AscendC::tiling::Mc2CcTiling tiling;
    CustomTiling param;
};
```

在所有核上创建HCCL对象，并调用InitV2接口初始化：

```
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);

    GM_ADDR contextGM = AscendC::GetHcclContext<0>();
    Hccl hccl;
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(UserCustomTilingData, tiling));

    // 调用HCCL的Prepare、Commit、Wait、Finalize接口
}
```

- [MatmulAllReduce 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_950.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md)
- [MatmulAlltoAll A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all.md)
- [MatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all_950.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
- [QuantMatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_allto_all_950.md)
- [WeightQuantMatmulAllReduce 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/weight_quant_matmul_all_reduce_950.md)
