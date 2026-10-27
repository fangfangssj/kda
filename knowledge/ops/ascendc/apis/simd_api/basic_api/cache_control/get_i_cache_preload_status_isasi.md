---
type: API
title: GetICachePreloadStatus
description: 调试接口，在ICachePreLoad后调用获取ICache预加载状态，返回0表示完成，返回1表示未完成。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cache_control/GetICachePreloadStatus_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_cache_intf.h
---

# GetICachePreloadStatus\(ISASI\)<a name="ZH-CN_TOPIC_0000001788032978"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_cache_intf.h"`。

GetICachePreloadStatus为调试接口，在ICachePreLoad后调用，用于获取ICache的PreLoad的状态：当返回值为0时，说明ICache的PreLoad已完成；当返回值为1时，说明ICache的PreLoad未完成。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline int64_t GetICachePreloadStatus()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

int64\_t类型，0表示ICache的PreLoad已完成，1表示ICache的PreLoad未完成。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```cpp
// 获取ICache预加载状态，0表示已完成，1表示未完成。
int64_t cachePreloadStatus = AscendC::GetICachePreloadStatus();
```
