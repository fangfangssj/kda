---
type: API
title: Clock
description: SIMD和SIMT调试场景中提供Clock时间戳功能，记录从程序启动到调用时刻所经历的时钟周期数。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/tuning_interface/clock.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/utils/debug/asc_time.h
---

# clock

## 功能说明

本接口在SIMD和SIMT调试场景中提供Clock时间戳功能，用于记录从程序启动到接口调用时刻所经历的时钟周期数（Cycle Count），便于精确分析执行延迟和性能瓶颈。

## 函数原型

```cpp
__aicore__ inline uint64_t clock(void)
```


以下接口为SIMT中所使用的clock接口，仅支持Ascend 950PR&950DT系列产品。

```cpp
inline uint64_t clock(void)
```


## 参数说明

无

## 返回值说明

从程序开始到调用时所经历的时钟周期数。

## 约束说明


-   SIMT场景的支持情况如下：
      
    - Ascend 950PR&950DT系列产品：支持。
      
      
    - Atlas A3系列产品：不支持。
      
      
    - Atlas A2系列产品：不支持。
      


## 需要包含的头文件

使用该接口需要包含"utils/debug/asc\_time.h"头文件。

```cpp
#include "utils/debug/asc_time.h"
```

## 调用示例

-   SIMT编程场景：

    ```cpp
    __global__ __launch_bounds__(1024) void SimtKernel(uint64_t* dst)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = clock();
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    __simt_vf__ __launch_bounds__(1024) inline void SimtKernel(__gm__ uint64_t* dst)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = clock();
    }
    ```

-   SIMD编程场景：

    ```cpp
    __global__ __aicore__ void AicoreKernel(__gm__ uint64_t* dst)
    {
        int idx = AscendC::GetBlockIdx();
        dst[idx] = clock();
    }
    ```
