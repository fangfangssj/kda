---
type: API
title: Grid协作组size
description: 获取当前Grid内的SIMT线程总数，与接口num_threads功能相同。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/grid_group/size.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# size

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前Grid内的SIMT线程总数，与接口[num_threads](num_threads.md)功能相同。

## 函数原型

```cpp
static unsigned long long size()
```

## 参数说明

无

## 返回值说明

当前Grid内的SIMT线程总数。

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
        unsigned long long thread_num = g.size();
        ...
    }
    ```
