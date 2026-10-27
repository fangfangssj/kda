---
type: API
title: NumThreads
description: 获取coalesced_group组内线程总数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/num_threads.md
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

获取`coalesced_group`组内线程总数。

## 函数原型

```cpp
unsigned int num_threads() const
```

## 参数说明

无

## 返回值说明

`coalesced_group`组内线程总数。

## 约束说明

无

## 调用示例

示例代码中偶数id的线程使用`coalesced_group`进行协同，奇数线程独立执行业务，一个Warp中共有16个偶数id的线程。

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            unsigned int thread_num = active.num_threads(); // 返回16
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
            unsigned int thread_num = active.num_threads(); // 返回16
        }
        ...
    }
    ```
