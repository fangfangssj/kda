---
type: API
title: Laneid
description: 获取线程在其所在Warp内的索引。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.programming_model.cooperative_groups
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/Warp_functions/lane_id_functions/laneid.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
- include/simt_api/device_warp_functions.h
- include/simt_api/device_atomic_functions.h
---

# laneid

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取线程在其所在Warp内的索引。

## 函数原型

```cpp
int32_t laneid()
```

## 参数说明

无

## 返回值说明

线程在其Warp内的索引，当前一个Warp中的线程数量为固定值32，故Lane ID的取值范围为[0,31]。

## 约束说明

无

## 调用示例

下面示例使用`laneid()`计算全部输入数据的和，并把结果赋值到输出数据的第0位置上，示例中使用[asc_reduce_add](../warp_reduce_functions/asc_reduce_add.md)，需另外包含`simt_api/device_warp_functions.h`头文件，使用[asc_atomic_add](../../atomic_operations/asc_atomic_add.md)，需另外包含`simt_api/device_atomic_functions.h`头文件。

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"
    #include "simt_api/device_atomic_functions.h"

    __global__ __launch_bounds__(1024) void kernel_laneid(int32_t* dst, int32_t* src, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        int32_t result = asc_reduce_add(src[idx]);
        if (laneid() == 0) {
            asc_atomic_add(dst, result);
        }
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"
    #include "simt_api/device_warp_functions.h"
    #include "simt_api/device_atomic_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_laneid(__gm__ int32_t* dst, __gm__ int32_t* src, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        int32_t result = asc_reduce_add(src[idx]);
        if (laneid() == 0) {
            asc_atomic_add(dst, result);
        }
    }
    ```
