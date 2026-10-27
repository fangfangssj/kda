---
type: API
title: Fmodf
description: 获取输入数据x除以y的余数。求余数时，商取x除以y浮点数结果的整数部分。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/fmodf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# fmodf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据x除以y的余数。求余数时，商取x除以y浮点数结果的整数部分。

## 函数原型

```cpp
inline float fmodf(float x, float y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据x除以y的余数。

-   x为0时，y不为0时，返回值为0。
-   x为有限数，y为inf或-inf时，返回值为x。
-   x为inf或-inf，或y为0时，返回值为nan。
-   x，y任意一个为nan时，返回值为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_fmodf(float *result, const float *x, const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = fmodf(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_fmodf_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = fmodf(x[idx], y[idx]);
    }

    __global__ __vector__ void run_fmodf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        asc_vf_call<compute_fmodf_vf>(dim3(256), result, x, y, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
y: 1.25, 2.25, 3.25, 4.25
result: 0.25, 0.75, 1.25, 1.75
```

- [GridSampler3D 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/grid_sampler_3d_950.md)
