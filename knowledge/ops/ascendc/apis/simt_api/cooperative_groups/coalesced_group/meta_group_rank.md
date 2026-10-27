---
type: API
title: MetaGroupRank
description: 获取当前线程所在的组在其父组划分出的子组集合中的排名，排名从0开始。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/meta_group_rank.md
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

获取当前线程所在的组在其父组划分出的子组集合中的排名，排名从0开始。

## 函数原型

```cpp
unsigned int meta_group_rank() const
```

## 参数说明

无

## 返回值说明

当前线程所在的组在其父组划分出的子组集合中的排名。

- 如果该组是通过`coalesced_threads`创建的，则`meta_group_rank()`的返回值为0。

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
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned int group_rank = active.meta_group_rank(); // 返回0
        }
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
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned int group_rank = active.meta_group_rank(); // 返回0
        }
        ...
    }
    ```
