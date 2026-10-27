---
type: API
title: asc_all
description: 判断是否所有活跃线程的输入均不为0。 当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数predicate进行判断，所有活跃线程的predicate均不为0，返回1，否则返回0。Warp内所有活跃线程返回相同的结果。
tags:
- task.lookup
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/Warp_functions/Warp_vote_functions/asc_all.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_warp_functions.h
---

# asc\_all

## 功能说明

头文件路径为：`"simt_api/device_warp_functions.h"`。

判断是否所有活跃线程的输入均不为0。

当Warp内所有活跃线程执行本接口后，对所有活跃线程的输入操作数`predicate`进行判断，所有活跃线程的`predicate`均不为0，返回1，否则返回0。Warp内所有活跃线程返回相同的结果。

## 函数原型

```cpp
inline int32_t asc_all(int32_t predicate)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

当Warp内所有活跃线程的输入均不为0，返回1，否则返回0。

## 约束说明

无

## 调用示例

以下样例比较两组输入数据，并由Lane 0写出当前Warp负责的数据是否全部相等。

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_warp_functions.h"

    __global__ __launch_bounds__(1024) void KernelAll(
        const int32_t* input1, const int32_t* input2, int32_t* equal_flags, uint64_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }

        uint32_t lane_id = threadIdx.x % warpSize;
        uint32_t warp_id = idx / warpSize;
        int32_t all_equal = asc_all(input1[idx] == input2[idx]);
        if (lane_id == 0) {
            equal_flags[warp_id] = all_equal;
        }
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_warp_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelAll(
        __gm__ const int32_t* input1, __gm__ const int32_t* input2, __gm__ int32_t* equal_flags,
        uint64_t total_length)
    {
        // asc_vf_call参数：dim3{1024, 1, 1}
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }

        uint32_t lane_id = threadIdx.x % warpSize;
        uint32_t warp_id = idx / warpSize;
        int32_t all_equal = asc_all(input1[idx] == input2[idx]);
        if (lane_id == 0) {
            equal_flags[warp_id] = all_equal;
        }
    }
    ```
