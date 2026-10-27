---
type: API
title: ThreadBlockTile协作组sync
description: 同步当前线程所属的thread_block_tile组内的所有线程。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/sync.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# sync

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

同步当前线程所属的`thread_block_tile`组内的所有线程，所有线程都执行到该同步点位置才能继续执行。

## 函数原型

```cpp
void sync() const
```

## 参数说明

无

## 返回值说明

无

## 约束说明

必须保证`thread_block_tile`组内所有线程都能执行到同一个`sync()`调用，否则是未定义行为。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        tile4.sync();
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        tile4.sync();
        ...
    }
    ```
