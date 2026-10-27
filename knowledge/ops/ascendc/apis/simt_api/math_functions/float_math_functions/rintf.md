---
type: API
title: Rintf
description: 获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/rintf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# rintf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取与输入数据最接近的整数，若存在两个同样接近的整数，则获取其中的偶数。

## 函数原型

```cpp
inline float rintf(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

与输入浮点数最接近的整数值。特别场景说明如下：

-   当`x`为0时，返回值为0。
-   当`x`为0.5时，返回值为0。
-   当`x`为1.5时，返回值为2。
-   当`x`为nan时，返回值为nan。
-   当`x`为inf时，返回值为inf。
-   当`x`为-inf时，返回值为-inf。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_rintf(float *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = rintf(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_rintf_vf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = rintf(x[idx]);
    }

    __global__ __vector__ void run_rintf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_rintf_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 0, 1, 1, 2
```

- [GridSampler3D 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/grid_sampler_3d_950.md)
- [ForeachRoundOffNumber 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_round_off_number_950.md)
