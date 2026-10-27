---
type: API
title: AllocMutexID
description: 从框架获取并占用一个MutexID，与ReleaseMutexID配合使用，获取的MutexID可传入Mutex::Lock/Unlock接口用于流水同步。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/AllocMutexID_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_common.h
---

# AllocMutexID \(ISASI\)<a name="ZH-CN_TOPIC_0000002352639064"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

从框架获取并占用一个MutexID，与[ReleaseMutexID](release_mutex_id_isasi.md)配合使用，管理MutexID的获取和释放。获取的MutexID可以传入[Mutex::Lock](lock.md)/[Mutex::Unlock](unlock.md)接口，用于核内异步流水同步。通过本接口获取的MutexID会纳入框架的MutexID分配记录，适用于`Mutex::Lock`/`Mutex::Unlock`与TQue等资源管理接口配合使用的场景，自行管理MutexID与资源管理接口配合使用时的限制请参见[Lock约束说明](lock.md#mutexid_resource_management_constraint)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline MutexID AllocMutexID()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

返回MutexID，其类型定义如下，每个ID表示一个Mutex锁。

```cpp
using MutexID = uint8_t;
```

## 约束说明<a name="section633mcpsimp"></a>

MutexID有数量限制，使用结束后应立即调用[ReleaseMutexID](release_mutex_id_isasi.md)释放，防止MutexID耗尽。

## 调用示例<a name="section837496171220"></a>

示例请参考[Lock调用示例](lock.md#section123275308128)。
