---
type: API
title: ReleaseMutexID
description: 从框架释放一个MutexID，与AllocMutexID配合使用，防止MutexID耗尽。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/ReleaseMutexID_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_common.h
---

# ReleaseMutexID \(ISASI\)<a name="ZH-CN_TOPIC_0000002406521145"></a>

## 功能说明<a name="section1386110162210"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

从框架释放一个MutexID，与[AllocMutexID](alloc_mutex_id_isasi.md)配合使用。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void ReleaseMutexID(MutexID id)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| id | 输入 | MutexID类型，应传入调用AllocMutexID获得的MutexID。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

MutexID有数量限制，使用结束后应立即调用[ReleaseMutexID](release_mutex_id_isasi.md)释放，防止MutexID耗尽。

## 调用示例<a name="section837496171220"></a>

示例请参考[Lock调用示例](lock.md#section123275308128)。
