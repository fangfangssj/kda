---
type: API
title: Isfinite
description: 判断浮点数是否为有限数（非inf、非nan）。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/isfinite.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# isfinite

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

判断浮点数是否为有限数（非inf、非nan）。

## 函数原型

```cpp
inline bool isfinite(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

-   false：输入为nan、inf、-inf。
-   true：输入为有限数。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_isfinite(bool *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = isfinite(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_isfinite_vf(__gm__ bool *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = isfinite(x[idx]);
    }

    __global__ __vector__ void run_isfinite(__gm__ bool *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_isfinite_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 1, 1, 1, 1
```

- [GridSampler3D 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/grid_sampler_3d_950.md)
- [RmsNorm 950 Direct Invoke 算子知识](../../../../operators/direct/nn/norm/rms_norm_950.md)
