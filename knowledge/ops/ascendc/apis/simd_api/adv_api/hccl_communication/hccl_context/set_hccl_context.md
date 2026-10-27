---
type: API
title: SetHcclContext
description: 设置通算融合算子每个通信域对应的context（消息区）地址。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL-Context/SetHcclContext.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
- 310p
---

# SetHcclContext

## 功能说明

设置通算融合算子每个通信域对应的context（消息区）地址。

## 函数原型

```
template <uint32_t index>
__aicore__ inline void SetHcclContext(__gm__ uint8_t* context)
```

## 参数说明

**表1**  参数说明

| 参数名 | 描述 |
| --- | --- |
| index | 模板参数，用来表示要设置的通信域ID，当前只支持2个通信域，index只能为0/1。 |
| context | 对应通信域的context（消息区）地址。 |

## 返回值说明

无

## 约束说明

当前最多只支持2个通信域。

## 调用示例

```
// 给GROUP_0设置消息区地址
AscendC::SetHcclContext<0>(contextGM);
```
