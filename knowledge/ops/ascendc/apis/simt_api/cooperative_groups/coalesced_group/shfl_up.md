---
type: API
title: ShflUp
description: 获取coalesced_group组内当前线程向前偏移delta的线程的数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/coalesced_group/shfl_up.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# shfl_up

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

获取`coalesced_group`组内当前线程向前偏移`delta`的线程的数据。

## 函数原型

```cpp
template <typename T>
T shfl_up(T var, unsigned int delta) const
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。支持的数据类型为：half、int32_t、uint32_t、float、half2、int64_t、uint64_t、bfloat16_t、bfloat16x2_t。 |
| delta | 输入 | 期望获取的`var`值所在线程在组内相对当前线程的向前偏移值。 |

## 返回值说明

`coalesced_group`组内当前线程向前偏移`delta`的线程输入的`var`值。若偏移后超出组范围，返回当前线程输入的`var`值。

## 约束说明

无

## 调用示例

示例代码中的条件分支将一个warp中所有线程id是偶数的线程组成`coalesced_group`协作组，组内各线程`shfl_up`接口返回结果如下图所示。

**图1**  shfl_up结果示意图
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/coalesced_group_shfl_up.png "shfl_up结果示意图")

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        if (threadIdx.x % 2 == 0) {
            coalesced_group active = coalesced_threads();
            uint32_t result = active.shfl_up(threadIdx.x + 100, 2);
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
            uint32_t result = active.shfl_up(threadIdx.x + 100, 2);
        }
        ...
    }
    ```
