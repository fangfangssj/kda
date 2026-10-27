---
type: API
title: ThreadBlockTile协作组shfl
description: thread_block_tile组内线程的数据交换接口，用于获取组内指定rank线程输入的var值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/shfl.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# shfl

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

`thread_block_tile`组内线程的数据交换接口，用于获取组内指定rank线程输入的`var`值。

## 函数原型

```cpp
template <typename T>
T shfl(T var, int src_rank) const
```

## 参数说明


**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。支持的数据类型为：half、int32_t、uint32_t、float、half2、int64_t、uint64_t、bfloat16_t、bfloat16x2_t。 |
| src_rank | 输入 | 期望获取的`var`值所在的线程在组内的排名。当`src_rank`大于等于组内线程数时，获取`src_rank % Size`对应rank线程的`var`值。 |

## 返回值说明

`thread_block_tile`组内指定线程输入的`var`值。

## 约束说明

- 在跨Warp的`thread_block_tile`场景（Size>32）下
    - 必须保证`thread_block_tile`组内线程指定相同的`src_rank`，否则是未定义行为。
    - 必须保证`thread_block_tile`组内线程均处于活跃状态，否则是未定义行为。

## 调用示例

示例代码中以4个线程为一组划分线程块，获取组内排名为2的线程输入的`var`值。

**图1**   shfl接口返回结果示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/thread_block_tile_shfl.png "thread_block_tile_shfl")

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.shfl(threadIdx.x + 100, 2);
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
        uint32_t result = tile4.shfl(threadIdx.x + 100, 2);
        ...
    }
    ```
