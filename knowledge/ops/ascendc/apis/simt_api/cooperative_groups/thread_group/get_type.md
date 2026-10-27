---
type: API
title: get_type
description: 头文件路径为："simt_api/cooperative_groups.h"。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_group/get_type.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# get_type

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取`thread_group`对象表示的协作组类型。

## 函数原型

```cpp
group_type get_type() const
```

## 参数说明

无

## 返回值说明

返回协作组类型，可能的枚举值如下：

| 枚举值 | 描述 |
| --- | --- |
| `group_type::thread_block_type` | thread_block协作组。 |
| `group_type::tiled_group_type` | thread_block_tile协作组。 |
| `group_type::coalesced_group_type` | coalesced_group协作组。 |
| `group_type::grid_group_type` | grid_group协作组，仅SIMT编程场景支持。 |

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
        group_type type = tile.get_type(); // 返回group_type::tiled_group_type
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
        group_type type = tile.get_type(); // 返回group_type::tiled_group_type
        ...
    }
    ```
