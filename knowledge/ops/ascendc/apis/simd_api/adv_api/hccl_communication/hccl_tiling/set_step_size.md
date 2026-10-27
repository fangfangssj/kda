---
type: API
title: SetStepSize
description: 设置细粒度通信时，通信算法的步长，即设置细粒度通信时，一次子通信任务执行或准备执行的通信算法的步骤数。例如，图1使用pairwise算法的AlltoAllV通信步骤示意图中，该细粒度通信场景下，AlltoAllV通信任务的通信步长为1。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetStepSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a3
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetStepSize

## 功能说明

设置[细粒度通信](../hccl_kernel/iterate.md)时，通信算法的步长，即设置细粒度通信时，一次子通信任务执行或准备执行的通信算法的步骤数。例如，[图1使用pairwise算法的AlltoAllV通信步骤示意图](../hccl_kernel/iterate.md)中，该细粒度通信场景下，AlltoAllV通信任务的通信步长为1。

## 函数原型

```
uint32_t SetStepSize(uint8_t stepSize)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| stepSize | 输入 | 设置细粒度通信时，每次通信的步长。0表示当前非细粒度通信。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明


目前仅支持Atlas A3系列产品。


## 调用示例

```
static ge::graphStatus AllToAllVCustomTilingFunc(gert::TilingContext* context)
{
    AllToAllVCustomV3TilingData* tiling = context->GetTilingData<AllToAllVCustomV3TilingData>();
    const std::string groupName = "testGroup";
    const std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_ALLTOALLV, algConfig, 0);
    mc2CcTilingConfig.SetStepSize(1U);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
```
