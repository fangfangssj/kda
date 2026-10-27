---
type: API
title: GetQueueNum
description: 获取BatchWrite通信任务的通信队列数量。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/GetQueueNum.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# GetQueueNum

## 功能说明

获取BatchWrite通信任务的通信队列数量。

## 函数原型

```
__aicore__ inline uint16_t GetQueueNum()
```

## 参数说明

无

## 返回值说明

BatchWrite通信任务的通信队列数量。

## 约束说明

无

## 调用示例

请参见[BatchWrite的调用示例](batch_write.md#调用示例)。

