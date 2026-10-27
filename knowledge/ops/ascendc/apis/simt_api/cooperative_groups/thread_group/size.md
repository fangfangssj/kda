---
type: API
title: size
description: 头文件路径为："simt_api/cooperative_groups.h"。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_group/size.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# size

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前`thread_group`组内的线程总数，与接口[num_threads()](num_threads.md)功能相同。

## 函数原型

```cpp
unsigned long long size() const
```

## 参数说明

无

## 返回值说明

当前`thread_group`组内的线程总数。

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
        unsigned long long thread_num = tile.size(); // 返回16
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
        unsigned long long thread_num = tile.size(); // 返回16
        ...
    }
    ```
