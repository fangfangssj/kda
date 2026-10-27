---
type: API
title: Ballot
description: 判断thread_block_tile组内每个线程的输入是否非零，返回32位无符号整数表示各线程对应bit位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/ballot.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# ballot

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

判断`thread_block_tile`组内每个线程的输入是否非零。

## 函数原型

```cpp
unsigned int ballot(int predicate) const
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

32-bit的无符号整数：若`thread_block_tile`组内线程输入的`predicate`不为0，则返回值中与线程rank对应的bit位为1，否则为0。

## 约束说明

仅支持`thread_block_tile`的模板参数`Size`小于等于32的场景，即仅支持单Warp内线程子组。

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
        uint32_t result = tile4.ballot(1);          // 返回0xf
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
        uint32_t result = tile4.ballot(1);          // 返回0xf
        ...
    }
    ```
