---
type: API
title: CoalescedGroup协作组size
description: 获取coalesced_group组内线程总数，与接口num_threads功能相同。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/size.md
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

获取`coalesced_group`组内线程总数，与接口[num_threads](num_threads.md)功能相同。

## 函数原型

```cpp
unsigned int size() const
```

## 参数说明

无

## 返回值说明

`coalesced_group`组内线程总数。

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
            unsigned int thread_num = active.size(); // 返回16
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
            unsigned int thread_num = active.size(); // 返回16
        }
        ...
    }
    ```
