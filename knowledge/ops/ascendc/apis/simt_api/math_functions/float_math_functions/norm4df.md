---
type: API
title: Norm4df
description: 获取输入数据a、b、c、d平方和的平方根。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/norm4df.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# norm4df

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据a、b、c、d的平方和（a^2 + b^2 + c^2 + d^2）的平方根。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002484776408.png)

## 函数原型

```cpp
inline float norm4df(float a, float b, float c, float d)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| a | 输入 | 源操作数。 |
| b | 输入 | 源操作数。 |
| c | 输入 | 源操作数。 |
| d | 输入 | 源操作数。 |

## 返回值说明

a^2 + b^2 + c^2 + d^2的平方根。

-   若a^2 + b^2 + c^2 + d^2的平方根超出float最大范围，返回值为inf。
-   若a, b, c, d任意一个或多个为±inf，返回值为inf。
-   若a, b, c, d任意一个或多个为nan同时不是±inf，返回值为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_norm4df(float *result, const float *x, const float *y, const float *z, const float *w, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = norm4df(x[idx], y[idx], z[idx], w[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_norm4df_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, __gm__ const float *z, __gm__ const float *w, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = norm4df(x[idx], y[idx], z[idx], w[idx]);
    }

    __global__ __vector__ void run_norm4df(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, __gm__ const float *z, __gm__ const float *w, uint32_t count)
    {
        asc_vf_call<compute_norm4df_vf>(dim3(256), result, x, y, z, w, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
y: 1.5, 2.5, 3.5, 4.5
z: -0.5, 0.5, 1.5, 2.5
w: 2, 3, 4, 5
result: 2.561738, 4.007805, 5.662376, 7.386643
```
