---
type: API
title: Query（HCCL集合通信）
description: 查询handleId对应的通信任务已经完成的轮次，最多返回repeat轮。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/Query.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# Query

## 功能说明

查询handleId对应的通信任务已经完成的轮次，最多返回repeat轮。该接口默认在所有核上工作，用户也可以在调用前通过[GetBlockIdx](../../../basic_api/tool_interface/system_resources_and_variables/get_block_idx.md)指定其在某一个核上运行。

## 函数原型

```
__aicore__ inline int32_t Query(HcclHandle handleId)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| handleId | 输入 | 对应通信任务的标识ID，只能使用Prepare原语接口的返回值。<br>using HcclHandle = int8_t; |

## 返回值说明

-   返回handleId对应的通信任务已执行的次数，最大值为repeat。
-   当执行异常时，返回-1。

## 约束说明

-   调用本接口前确保已调用过[InitV2](init_v2.md)和[SetCcTilingV2](set_cc_tiling_v2.md)接口。
-   入参handleId只能使用Prepare原语对应接口的返回值。
-   本接口在AIC核或者AIV核上调用必须与对应的Prepare接口的调用核保持一致。

## 调用示例

```
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
hccl.InitV2(contextGM, &tilingData);
auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, mc2CcTiling));
if (ret != HCCL_SUCCESS) {
    return;
}
if (AscendC::g_coreType == AIC) {
    auto repeat = 10;
    HcclHandle handleId = hccl.ReduceScatter(
        sendBuf, recvBuf, 100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, repeat);
    hccl.Commit(handleId); // 通知服务端可以执行上述的ReduceScatter任务
    int32_t finishedCount = hccl.Query(handleId);
    while (hccl.Query(handleId) < repeat) {
    }                // 等待查询到handleId对应的通信任务执行repeat次
    hccl.Finalize(); // 后续无其他通信任务，通知服务端执行上述ReduceScatter任务之后即可以退出
}
```
