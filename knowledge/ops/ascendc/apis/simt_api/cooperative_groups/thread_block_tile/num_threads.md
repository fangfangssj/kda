---
type: API
title: NumThreads
description: 获取当前线程所属的thread_block_tile组内的线程总数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/num_threads.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# num_threads

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程所属的`thread_block_tile`组内的线程总数。

## 函数原型

```cpp
unsigned int num_threads() const
```

## 参数说明

无

## 返回值说明

当前线程所属的`thread_block_tile`组内的线程总数。

## 约束说明

无

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
        unsigned int thread_num = tile4.num_threads();    // 返回4
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
        unsigned int thread_num = tile4.num_threads();    // 返回4
        ...
    }
    ```
