---
type: API
title: Log1pf
description: 获取以e为底输入数据加1的对数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/log1pf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# log1pf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取以e为底，输入数据加1的对数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002484776382.png)

## 函数原型

```cpp
inline float log1pf(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

以e为底的x+1的对数。

-   当x为nan时，返回值为nan。
-   当x为-1时，返回值为-inf。
-   当x小于-1时，返回值为nan。
-   当x为+inf时，返回值为+inf。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_log1pf(float *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = log1pf(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_log1pf_vf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = log1pf(x[idx]);
    }

    __global__ __vector__ void run_log1pf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_log1pf_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 1.25, 2.25, 3.25
result: 0.2231435, 0.8109302, 1.178655, 1.446919
```

- [ForeachLog1p 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_log1p_950.md)
