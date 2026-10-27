---
type: API
title: BlockIndex
description: 获取当前线程所属线程块在Grid中的三维索引。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/grid_group/block_index.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# block_index

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程所属线程块在Grid中的三维索引。

## 函数原型

```cpp
static dim3 block_index()
```

## 参数说明

无

## 返回值说明

当前线程所属线程块在Grid中的三维索引。

## 约束说明

仅支持SIMT编程场景，不支持SIMD与SIMT混合编程场景。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        grid_group g = this_grid();
        dim3 block = g.block_index();
        ...
    }
    ```
