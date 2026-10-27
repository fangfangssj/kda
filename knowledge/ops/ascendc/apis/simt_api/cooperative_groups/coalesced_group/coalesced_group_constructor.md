---
type: API
title: coalesced_group构造函数
description: coalesced_group不提供默认的构造函数，用户使用coalesced_threads函数创建coalesced_group对象。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/coalesced_group_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# coalesced_group构造函数

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

`coalesced_group`不提供默认的构造函数，用户使用`coalesced_threads`函数创建`coalesced_group`对象。

## 函数原型

```cpp
coalesced_group coalesced_threads()
```

## 参数说明

无

## 返回值说明

返回一个`coalesced_group`对象，该对象包含当前Warp内所有活跃的线程。

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
            coalesced_group active = coalesced_threads(); // 该coalesced_group中包含当前warp内所有线程id为偶数的线程
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
            coalesced_group active = coalesced_threads(); // 该coalesced_group中包含当前warp内所有线程id为偶数的线程
        }
        ...
    }
    ```
