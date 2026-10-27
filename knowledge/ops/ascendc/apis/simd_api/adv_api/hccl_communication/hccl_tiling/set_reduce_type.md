---
type: API
title: SetReduceType
description: 设置Reduce操作类型，仅对有归约操作的通信任务生效。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetReduceType.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetReduceType

## 功能说明

设置Reduce操作类型，仅对有归约操作的通信任务生效。

## 函数原型

```
uint32_t SetReduceType(uint32_t reduceType, uint8_t dstDataType = 0, uint8_t srcDataType = 0)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| reduceType | 输入 | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见[表2](../hccl_kernel/hccl_usage.md#table2469980529)参数说明。 |
| dstDataType | 输入 | 通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../hccl_kernel/hccl_usage.md#table116710585514)。 |
| srcDataType | 输入 | 通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../hccl_kernel/hccl_usage.md#table116710585514)。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明


对于Ascend 950PR&950DT系列产品，AI CPU作为服务端时，该接口暂不支持使用。


对于Atlas A3系列产品，该接口暂不支持使用。


对于Atlas A2系列产品，该接口暂不支持使用。


## 调用示例

本接口的调用示例请见[调用示例](set_op_type.md#调用示例)。
