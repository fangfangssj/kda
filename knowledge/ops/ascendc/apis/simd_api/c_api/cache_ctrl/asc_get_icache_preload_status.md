---
type: API
title: AscGetIcachePreloadStatus
description: 调试接口，在asc_icache_preload后调用获取ICache预加载状态，返回0表示完成，1表示未完成。
tags:
- task.lookup
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_get_icache_preload_status.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/cache_ctrl/cache_ctrl.h
---

# asc_get_icache_preload_status

## 功能说明

头文件路径为：`"c_api/cache_ctrl/cache_ctrl.h"`。

该接口为调试接口，在[asc_icache_preload](asc_icache_preload.md)后调用，用于获取ICache的PreLoad的状态：

- 当返回值为0时，说明ICache的PreLoad已完成。
- 当返回值为1时，说明ICache的PreLoad未完成。

## 函数原型

```c
__aicore__ inline int64_t asc_get_icache_preload_status()
```

## 参数说明

无

## 返回值说明

返回`int64_t`类型的ICache预取状态值，仅有效位取值如下：

- 返回`0`：ICache预取空闲，上一次预取已完成。
- 返回`1`：ICache预取正在进行中，尚未完成。

## 流水类型

PIPE_S

## 约束说明

本接口仅用于调试与性能分析场景下的状态查询，不影响预取本身的执行流程。

## 调用示例

本接口需在调用[asc_icache_preload](asc_icache_preload.md)后使用，完整调用示例请参考[asc_icache_preload的调用示例](asc_icache_preload.md#调用示例)。
