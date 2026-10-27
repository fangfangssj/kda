---
type: API
title: LanemaskGt
description: 获取当前线程的32位掩码，在所属Warp中将Lane ID严格大于当前线程的线程对应位设为1，其余位为0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/Warp_functions/lane_id_functions/lanemask_gt.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
- include/simt_api/device_warp_functions.h
---

# lanemask\_gt

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取当前线程的一个32位掩码，在当前线程所属的Warp中，将“Lane ID严格大于当前线程”的线程的对应位设为1，其余位为0。

如Lane ID为0的线程，调用本接口获取到32位掩码：1111 1111 1111 1111 1111 1111 1111 1110。

如Lane ID为31的线程，调用本接口获取到32位掩码：0000 0000 0000 0000 0000 0000 0000 0000。

**图1**  lanemask\_gt示意图

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/lanemask_gt.png "lanemask_gt图示")

## 函数原型

```cpp
int32_t lanemask_gt()
```

## 参数说明

无

## 返回值说明

返回一个32位整数的位掩码。

## 约束说明

无

## 调用示例

下面示例使用`lanemask_gt()`统计当前Lane之后的线程数，并据此判断当前线程位于Warp的上半区还是下半区，再通过`asc_shfl()`分别广播上下半Warp中指定Lane的数据。示例中使用[asc_shfl](../warp_shfl_functions/asc_shfl.md)，需另外包含`simt_api/device_warp_functions.h`头文件，其中`srcLane`的取值范围为[0, 15]。

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"

    __global__ __launch_bounds__(1024) void kernel_lanemask_gt(int32_t* src, int32_t* dst, int32_t srcLane)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t lanes_after = __popc(static_cast<uint32_t>(lanemask_gt()));
        int32_t group_leader_lane = (lanes_after >= 16) ? srcLane : (srcLane + 16);
        int32_t value = src[idx];
        value = asc_shfl(value, group_leader_lane, 32);
        dst[idx] = value;
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"

    __simt_vf__ __launch_bounds__(1024) void kernel_lanemask_gt(__gm__ int32_t* src, __gm__ int32_t* dst, int32_t srcLane)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int32_t lanes_after = __popc(static_cast<uint32_t>(lanemask_gt()));
        int32_t group_leader_lane = (lanes_after >= 16) ? srcLane : (srcLane + 16);
        int32_t value = src[idx];
        value = asc_shfl(value, group_leader_lane, 32);
        dst[idx] = value;
    }
    ```
