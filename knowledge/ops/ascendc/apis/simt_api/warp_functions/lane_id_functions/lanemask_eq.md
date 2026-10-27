---
type: API
title: LanemaskEq
description: 获取当前线程的32位掩码，在所属Warp中只有当前线程所在的Lane位被置1，其余为0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/Warp_functions/lane_id_functions/lanemask_eq.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
- include/simt_api/device_warp_functions.h
---

# lanemask\_eq

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取当前线程的一个32位掩码，在当前线程所属的Warp中，只有当前线程所在的Lane位被置为1，其余位为0。

如Lane ID为0的线程，调用本接口获取到32位掩码：0000 0000 0000 0000 0000 0000 0000 0001。

如Lane ID为31的线程，调用本接口获取到32位掩码：1000 0000 0000 0000 0000 0000 0000 0000。

**图1**  lanemask\_eq示意图

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/lanemask_eq.png "lanemask_eq图示")

## 函数原型

```cpp
int32_t lanemask_eq()
```

## 参数说明

无

## 返回值说明

返回一个32位整数的位掩码。

## 约束说明

无

## 调用示例

下面示例使用`lanemask_eq()`获取当前Lane对应的位掩码，并根据掩码将Warp内偶数Lane和奇数Lane的数据分别重排到输出Warp的前16个位置和后16个位置。示例中使用[asc_shfl](../warp_shfl_functions/asc_shfl.md)，需另外包含`simt_api/device_warp_functions.h`头文件，其中`srcLane`的取值范围为[0, 15]。

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"

    __global__ __launch_bounds__(1024) void kernel_lanemask_eq(int32_t* dst, int32_t* src, int32_t srcLane)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t self_mask = lanemask_eq();
        int32_t group_leader_lane = ((self_mask & 0x0000FFFFU) != 0) ? srcLane : (srcLane + 16);
        int32_t value = src[idx];
        value = asc_shfl(value, group_leader_lane, 32);
        dst[idx] = value;
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"

    __simt_vf__ __launch_bounds__(1024) void kernel_lanemask_eq(__gm__ int32_t* dst, __gm__ int32_t* src, int32_t srcLane)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t self_mask = lanemask_eq();
        int32_t group_leader_lane = ((self_mask & 0x0000FFFFU) != 0) ? srcLane : (srcLane + 16);
        int32_t value = src[idx];
        value = asc_shfl(value, group_leader_lane, 32);
        dst[idx] = value;
    }
    ```
