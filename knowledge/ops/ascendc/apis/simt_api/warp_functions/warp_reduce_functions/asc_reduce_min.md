---
type: API
title: asc_reduce_min
description: 对Warp内所有活跃线程输入val求最小值。Warp内所有活跃线程返回相同的结果。
tags:
- task.lookup
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/Warp_functions/Warp_reduce_functions/asc_reduce_min.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_warp_functions.h
- include/simt_api/asc_fp16.h
---

# asc\_reduce\_min

## 功能说明

头文件路径为：`"simt_api/device_warp_functions.h"`（除half类型之外的接口）、`"simt_api/asc_fp16.h"`（half类型接口）。

对Warp内所有活跃线程输入val求最小值。Warp内所有活跃线程返回相同的结果。

## 函数原型

```cpp
inline int32_t asc_reduce_min(int32_t val)
```

```cpp
inline uint32_t asc_reduce_min(uint32_t val)
```

```cpp
inline float asc_reduce_min(float val)
```

```cpp
inline half asc_reduce_min(half val)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| val | 输入 | 源操作数。 |

## 返回值说明

Warp内所有活跃线程输入val的最小值。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_warp_functions.h"

    __global__ __launch_bounds__(1024) void KernelReduceMin(int32_t* dst)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t laneId = idx % 32;
        int32_t result = asc_reduce_min(laneId); // 返回值为0
        dst[idx] = result;
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_warp_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelReduceMin(__gm__ int32_t* dst)
    {
        // asc_vf_call参数：dim3{1024, 1, 1}
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t laneId = idx % 32;
        int32_t result = asc_reduce_min(laneId); // 返回值为0
        dst[idx] = result;
    }
    ```
