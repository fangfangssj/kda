---
type: API
title: SetQueueNum
description: 设置每个向服务端下发任务的核上的BatchWrite通信队列数量。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetQueueNum.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetQueueNum

## 功能说明

设置每个向服务端下发任务的核上的BatchWrite通信队列数量。

## 函数原型

```
uint32_t SetQueueNum(uint16_t num)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| num | 输入 | 表示队列的数量。参与通信的核数*队列数量支持设置的取值范围为[0, 40]，参与通信的核数的设置请参考[SetCommBlockNum](set_comm_block_num.md)。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明


本接口仅在Atlas A3系列产品上通信类型为HCCL\_CMD\_BATCH\_WRITE时生效。


## 调用示例

```
const char* groupName = "testGroup";
uint32_t opType = HCCL_CMD_BATCH_WRITE;
std::string algConfig = "BatchWrite=level0:fullmesh";
uint32_t reduceType = 0; // BatchWrite无归约语义，此处占位
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.SetQueueNum(2U);
```
