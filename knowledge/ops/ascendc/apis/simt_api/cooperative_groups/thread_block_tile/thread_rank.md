---
type: API
title: ThreadRank
description: 获取当前线程在所属的thread_block_tile组内的排名，排名从0开始。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/thread_rank.md
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

获取当前线程在所属的`thread_block_tile`组内的排名，排名从0开始。

## 函数原型

```cpp
unsigned int thread_rank() const
```

## 参数说明

无

## 返回值说明

当前线程在所属的`thread_block_tile`组内的排名。

## 约束说明

无

## 调用示例

示例代码中以4个线程为一组划分线程块，各线程在所属的`thread_block_tile`组内的排名如下图所示。

**图1**   thread_rank接口返回值示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/thread_block_tile_rank.png "thread_block_tile_rank")

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        unsigned int rank = tile4.thread_rank();
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
        unsigned int rank = tile4.thread_rank();
        ...
    }
    ```
