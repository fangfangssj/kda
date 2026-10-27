---
type: API
title: Unlock
description: 当前流水的前序指令执行完成后，根据MutexID释放对应Mutex，需与Lock接口严格成对使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Unlock.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_common.h
---

# Unlock<a name="ZH-CN_TOPIC_0000002406681009"></a>

## 功能说明<a name="section54681522111017"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

指定流水的前序指令执行完成后，根据MutexID释放对应Mutex。

> [!NOTE]说明
> Unlock接口需与[Lock](lock.md)接口配合使用，对同一个MutexID，Lock和Unlock必须严格成对出现。详细约束请参考[Lock约束说明](lock.md#section184751024101111)。

## 函数原型<a name="section1568410468104"></a>

```cpp
template <pipe_t pipe>
static __aicore__ inline void Unlock(MutexID id)
```

## 参数说明<a name="section74061251191017"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| pipe | 模板参数，表示流水类别。支持PIPE_S、PIPE_M、PIPE_V、PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX，各流水类型的含义请参考[硬件流水类型](intra_core_sync_overview.md#硬件流水类型)。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| id | 输入 | 进行流水同步管理的MutexID。在不同编程范式中，该ID的获取以及释放方式不同，详细情况参考[Lock约束说明](lock.md#section184751024101111)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section184751024101111"></a>

参考[Lock约束说明](lock.md#section184751024101111)。

## 调用示例<a name="section123275308128"></a>

Lock与Unlock必须严格配对使用，详细示例请参考[Lock调用示例](lock.md#section123275308128)。
