---
type: API
title: MetaGroupRank
description: 获取当前线程所在的组在从父组划分出的子组集合中的排名，排名从0开始。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/meta_group_rank.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# meta_group_rank

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程所在的组在从父组划分出的子组集合中的排名，排名从0开始。

## 函数原型

```cpp
unsigned int meta_group_rank() const
```

## 参数说明

无

## 返回值说明

当前线程所在的组在从父组划分出的子组集合中的排名。

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
        unsigned int group_rank = tile4.meta_group_rank();
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
        unsigned int group_rank = tile4.meta_group_rank();
        ...
    }
    ```
