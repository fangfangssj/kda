---
type: API
title: SetSkipBufferWindowCopy
description: 设置通信算法获取输入数据的位置。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/SetSkipBufferWindowCopy.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/hccl/hccl_tiling.h
---

# SetSkipBufferWindowCopy

## 功能说明

设置通信算法获取输入数据的位置。

## 函数原型

```
uint32_t SetSkipBufferWindowCopy(uint8_t skipBufferWindowCopy)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| skipBufferWindowCopy | 输入 | 表示通信算法获取输入数据的位置，uint8_t类型。 |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明

无

## 调用示例

本接口的调用示例请见[调用示例](set_op_type.md#调用示例)。
