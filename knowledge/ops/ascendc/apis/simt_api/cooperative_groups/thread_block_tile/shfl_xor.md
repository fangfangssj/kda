---
type: API
title: ShflXor
description: 将当前线程的rank与lane_mask进行按位异或运算得到的rank，获取该rank的线程输入的var值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block_tile/shfl_xor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/cooperative_groups.h
---

# shfl_xor

## 功能说明

头文件路径为：`"simt_api/cooperative_groups.h"`。

将当前线程的rank与`lane_mask`进行按位异或运算得到的rank，获取该rank的线程输入的`var`值。

## 函数原型

```cpp
template <typename T>
T shfl_xor(T var, unsigned int lane_mask) const
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| var | 输入 | 线程用于交换的输入操作数。支持的数据类型为：half、int32_t、uint32_t、float、half2、int64_t、uint64_t、bfloat16_t、bfloat16x2_t。 |
| lane_mask | 输入 | 与当前线程rank做异或运算的操作数。 |

## 返回值说明

`thread_block_tile`组内指定线程的`var`值。

## 约束说明

- 仅支持`thread_block_tile`的模板参数`Size`小于等于32的场景，即仅支持单Warp内线程子组。
- `lane_mask`必须小于`thread_block_tile`组内线程数。

## 调用示例

以4个线程为一组划分线程块，获取组内当前线程rank与`lane_mask`按位异或后对应rank线程输入的`var`值。

**图1**  shfl_xor接口返回结果示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/thread_block_tile_shfl_xor.png "thread_block_tile_shfl_xor")

- SIMT编程场景：

    ```cpp
    #include "simt_api/cooperative_groups.h"

    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.shfl_xor(threadIdx.x + 100, 1);
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
        uint32_t result = tile4.shfl_xor(threadIdx.x + 100, 1);
        ...
    }
    ```
