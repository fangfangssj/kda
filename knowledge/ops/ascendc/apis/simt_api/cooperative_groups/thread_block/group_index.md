---
type: API
title: GroupIndex
description: 获取当前线程所属的线程块在网格（grid）中的三维索引，返回值与内置变量blockIdx相同。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block/group_index.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# group_index

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程所属的线程块在网格（grid）中的三维索引，返回值与内置变量`blockIdx`相同。

## 函数原型

```cpp
static dim3 group_index()
```

## 参数说明

无

## 返回值说明

当前线程所属的线程块在网格中的三维索引。

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
        dim3 block_idx = g.group_index();
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
        dim3 block_idx = g.group_index();
        ...
    }
    ```