---
type: API
title: __trap
description: 在SIMT代码中调用此接口会中断算子的运行，适用于Kernel侧异常场景的调试。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/tuning_interface/__trap.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/utils/debug/asc_assert.h
---

# \_\_trap

## 功能说明

在SIMT代码中调用此接口会中断算子的运行，适用于核函数（Kernel）侧异常场景的调试。

## 函数原型

```cpp
inline void __trap()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 需要包含的头文件

使用该接口需要包含"utils/debug/asc\_assert.h"头文件。

```cpp
#include "utils/debug/asc_assert.h"
```

## 调用示例

-   SIMT编程场景：

    ```cpp
    __global__ __launch_bounds__(1024) inline void SimtKernel(float* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (isnan(x[idx])) {
            __trap();
        }
        dst[idx] = x[idx];
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    __simt_vf__ __launch_bounds__(1024) inline void SimtKernel(__gm__ float* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (isnan(x[idx])) {
            __trap();
        }
        dst[idx] = x[idx];
    }
    ```
