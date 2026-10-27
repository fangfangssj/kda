---
type: API
title: Ballot
description: 判断coalesced_group组内每个线程的输入是否非零。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/ballot.md
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

判断`coalesced_group`组内每个线程的输入是否非零。

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

32-bit的无符号整数：若`coalesced_group`组内线程输入的`predicate`不为0，则返回值中与线程`rank`对应的`bit`位为1，否则为0。

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
            uint32_t result = active.ballot(1); // 返回0xFFFF
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
            uint32_t result = active.ballot(1); // 返回0xFFFF
        }
        ...
    }
    ```
