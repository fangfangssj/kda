---
type: API
title: ThreadRank
description: 获取当前线程在线程块内所有线程中的排名，排名从0开始。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block/thread_rank.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# thread_rank

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程在线程块内所有线程中的排名，排名从0开始。

## 函数原型

```cpp
static unsigned int thread_rank()
```

## 参数说明

无

## 返回值说明

当前线程在线程块内所有线程中的排名（等价于`threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`）。

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
        thread_block g = this_thread_block();
        unsigned int rank = g.thread_rank();
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
        thread_block g = this_thread_block();
        unsigned int rank = g.thread_rank();
        ...
    }
    ```