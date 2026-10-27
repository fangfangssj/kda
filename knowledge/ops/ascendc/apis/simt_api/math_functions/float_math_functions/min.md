---
type: API
title: min
description: 获取两个输入数据中的最小值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/min.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# min

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取两个输入数据中的最小值。

## 函数原型

```cpp
float min(float x, float y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据中的最小值。

-   x为-0，y为+0时，返回x。
-   x为+0，y为-0时，返回y。
-   x为nan时，返回值为y。
-   y为nan时，返回值为x。
-   x，y同时为nan时，返回值为nan。
-   x，y任意一个为-inf时，返回值为-inf。
-   x为inf时，返回值为y。
-   y为inf时，返回值为x。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_min(float *result, const float *x, const float *y, uint32_t input_total_length)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_total_length) {
            return;
        }
        result[idx] = min(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_min_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t input_total_length)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= input_total_length) {
            return;
        }
        result[idx] = min(x[idx], y[idx]);
    }

    __global__ __vector__ void run_min(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t input_total_length)
    {
        asc_vf_call<compute_min_vf>(dim3(256), result, x, y, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: 0.25, 0.75, 1.25, 1.75
y: 1.5, 2.5, 3.5, 4.5
result: 0.25, 0.75, 1.25, 1.75
```
