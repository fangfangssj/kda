---
type: API
title: thread_rank
description: 头文件路径为："simt_api/cooperative_groups.h"。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_group/thread_rank.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# thread_rank

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程在所属`thread_group`组内的排名，排名从0开始计数。

## 函数原型

```cpp
unsigned long long thread_rank() const
```

## 参数说明

无

## 返回值说明

当前线程在所属`thread_group`组内的排名。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        thread_group tile = tiled_partition(this_thread_block(), 16);
        unsigned long long rank = tile.thread_rank(); // 取值范围为0到15
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        thread_group tile = tiled_partition(this_thread_block(), 16);
        unsigned long long rank = tile.thread_rank(); // 取值范围为0到15
        ...
    }
    ```
