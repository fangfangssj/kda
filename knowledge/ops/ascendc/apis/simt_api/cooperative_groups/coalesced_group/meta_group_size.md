---
type: API
title: MetaGroupSize
description: 获取当前线程所在的协作组其直接父组被划分时创建的子组的数量。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/meta_group_size.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# meta_group_size

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取当前线程所在协作组的直接父组被划分时创建的子组数量。

## 函数原型

```cpp
unsigned int meta_group_size() const
```

## 参数说明

无

## 返回值说明

直接父组被划分时创建的子组的数量。

- 如果该组是通过`coalesced_threads`创建的，则`meta_group_size()`的返回值将为1。

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
            unsigned int group_size = active.meta_group_size(); // 返回1
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
            unsigned int group_size = active.meta_group_size(); // 返回1
        }
        ...
    }
    ```
