---
type: API
title: QueueBarrier
description: 阻塞服务端上指定队列的BatchWrite通信任务，直到指定范围内所有队列上的任务完成执行，从而实现指定范围内队列的同步。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/QueueBarrier.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# QueueBarrier

## 功能说明

阻塞服务端上指定队列的BatchWrite通信任务，直到指定范围内所有队列上的任务完成执行，从而实现指定范围内队列的同步。

## 函数原型

```
template <ScopeType type = ScopeType::ALL>
__aicore__ inline void QueueBarrier(uint16_t queueID)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| type | 输入 | 表示阻塞服务端的通信任务范围。类型为ScopeType，定义如下代码所示，其中参数含义为：<br> ALL：阻塞所有队列上的通信任务<br>    QUEUE：暂不支持<br>    BLOCK：暂不支持<br>    INVALID_TYPE：暂不支持<br><br> 该参数的默认值为ScopeType::ALL。当前参数仅支持取值为ScopeType::ALL。|

```
enum class ScopeType : uint8_t { ALL, QUEUE, BLOCK, INVALID_TYPE };
```

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| queueID | 输入 | 表示队列ID。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

请参见[BatchWrite的调用示例](batch_write.md#调用示例)。
