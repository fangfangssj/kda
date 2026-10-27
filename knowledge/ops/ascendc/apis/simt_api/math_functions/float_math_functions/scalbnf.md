---
type: API
title: Scalbnf
description: 获取输入数据x与2的n次方的乘积。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/scalbnf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# scalbnf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据x与2的n次方的乘积。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002486622064.png)

## 函数原型

```cpp
inline float scalbnf(float x, int32_t n)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| n | 输入 | 源操作数。 |

## 返回值说明

输入数据x乘以2的n次幂的结果。

-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为-inf。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_scalbnf(float *result, const int *n, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = scalbnf(x[idx], n[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_scalbnf_vf(__gm__ float *result, __gm__ const int *n, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = scalbnf(x[idx], n[idx]);
    }

    __global__ __vector__ void run_scalbnf(__gm__ float *result, __gm__ const int *n, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_scalbnf_vf>(dim3(256), result, n, x, count);
    }
    ```

输入输出示例如下：

```
n: 1, 2, 3, 1
x: 0.25, 0.75, 1.25, 1.75
result: 0.5, 3, 10, 3.5
```
