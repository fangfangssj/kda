---
type: API
title: Modff
description: 将输入参数分解为小数部分和整数部分。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/modff.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# modff

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

将输入参数分解为小数部分和整数部分。

## 函数原型

```cpp
inline float modff(float x, float *n)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| n | 输出 | UB、Global Memory或栈空间的地址，用于存储输入数据的整数部分。 |

## 返回值说明

输入参数的小数部分。

-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为0。
-   当x为-inf时，返回值为0。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_modff(float *result, float *integer_part, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = modff(x[idx], integer_part + idx);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_modff_vf(__gm__ float *result, __gm__ float *integer_part, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = modff(x[idx], integer_part + idx);
    }

    __global__ __vector__ void run_modff(__gm__ float *result, __gm__ float *integer_part, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_modff_vf>(dim3(256), result, integer_part, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 0.25, 0.75, 0.25, 0.75
integer_part: 0 0 1 1
```
