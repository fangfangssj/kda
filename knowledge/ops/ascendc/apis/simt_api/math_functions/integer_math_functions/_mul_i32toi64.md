---
type: API
title: __mul_i32toi64
description: 计算输入32位整数x和y的乘积，返回64位结果。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__mul_i32toi64.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_mul\_i32toi64

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

计算输入32位整数x和y的乘积，返回64位结果。

## 函数原型

```cpp
long long __mul_i32toi64(int x, int y)
```

```cpp
unsigned long long __mul_i32toi64(unsigned int x, unsigned int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入数据x和y乘积的64位结果。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelMul_i32toi64(long long* dst, int* x, int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul_i32toi64(x[idx], y[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelMul_i32toi64(unsigned long long* dst, unsigned int* x, unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul_i32toi64(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelMul_i32toi64(__gm__ long long* dst, __gm__ int* x, __gm__ int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul_i32toi64(x[idx], y[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelMul_i32toi64(__gm__ unsigned long long* dst, __gm__ unsigned int* x, __gm__ unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul_i32toi64(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

- 当输入数据类型为`int`时：

    ```text
    x: -12, 23, -7, 6
    y: 6, 3, 4, 15
    result: -72, 69, -28, 90
    ```

- 当输入数据类型为`unsigned int`时：

    ```text
    x: 0, 12, 23, 70
    y: 6, 15, 3, 4
    result: 0, 180, 69, 280
    ```

- [SimThreadExponential 950 ACLNN 算子知识](../../../../operators/aclnn/math/random/sim_thread_exponential_950.md)
