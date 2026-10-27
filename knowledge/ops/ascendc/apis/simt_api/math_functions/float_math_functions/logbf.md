---
type: API
title: Logbf
description: 计算以2为底输入数据的对数，对结果向下取整返回浮点数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/logbf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# logbf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

计算以2为底，输入数据的对数，并对结果向下取整，返回浮点数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002516816343.png)

## 函数原型

```cpp
inline float logbf(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

以2为底的x的绝对值(|x|)的对数，并向下取整后的值。

-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为inf。
-   当x为nan时，返回值为nan。
-   当x为0时，返回值为-inf。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_logbf(float *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = logbf(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_logbf_vf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = logbf(x[idx]);
    }

    __global__ __vector__ void run_logbf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_logbf_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 1, 2, 3, 4
result: 0, 1, 1, 2
```
