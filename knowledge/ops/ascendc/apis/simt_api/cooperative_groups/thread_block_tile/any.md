---
type: API
title: ThreadBlockTile协作组any
description: 判断thread_block_tile组内是否有线程的输入不为0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/any.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# any

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

判断`thread_block_tile`组内是否有线程的输入不为0。

## 函数原型

```cpp
int any(int predicate) const
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

`thread_block_tile`组内所有线程的输入均为0，返回0，否则返回1。

## 约束说明

- 当`thread_block_tile`的模板参数`Size`大于32时，即`Size`为64、128、256、512、1024、2048的跨Warp协作组场景，必须保证`thread_block_tile`组内所有线程均处于活跃状态，否则是未定义行为。

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
        int result = tile4.any(1);                  // 返回1
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
        int result = tile4.any(1);                  // 返回1
        ...
    }
    ```
