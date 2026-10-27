---
type: API
title: DataCachePreload
description: 从GM地址预加载数据到DCache，每次调用预加载一个Cache Line大小的数据，提升后续访问效率。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cache_control/DataCachePreload.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_cache_intf.h
---

# DataCachePreload<a name="ZH-CN_TOPIC_0000001788032974"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_cache_intf.h"`。

从源地址所在的特定GM地址预加载数据到DCache中，每次调用只能预加载一个Cache Line大小的数据。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& src, const T cacheOffset)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| src | 输入 | 源操作数，类型为[GlobalTensor](../data_structures/global_tensor/global_tensor_intro.md)。支持的数据类型为：uint64_t。 |
| cacheOffset | 输入 | 在源操作数上偏移cacheOffset大小开始加载数据，单位为byte，支持的数据类型为：int16_t、int64_t。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

频繁调用此接口可能导致保留站（Reservation Station，用于暂存待执行指令的硬件队列）发生拥塞。在此情况下，该指令将被视为NOP指令，进而阻塞Scalar流水线。因此，建议连续调用此接口的次数不要超过4次。

## 调用示例<a name="section837496171220"></a>

```cpp
AscendC::GlobalTensor<uint64_t> srcGlobal;
int64_t cacheOffset = 0;
AscendC::DataCachePreload(srcGlobal, cacheOffset);
```
