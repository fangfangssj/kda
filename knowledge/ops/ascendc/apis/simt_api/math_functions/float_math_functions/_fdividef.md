---
type: API
title: __fdividef
description: 获取两个输入数据相除的结果。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/__fdividef.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_fdividef

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取两个输入数据相除的结果。

## 函数原型

```cpp
float __fdividef(float x, float y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据相除的结果。特殊值如下：

-   当x为正数，y为0.0时，返回值为+inf。
-   当x为负数，y为0.0时，返回值为-inf。
-   当x为0.0，y为0.0时，返回值为nan。
-   当x、y任意一个为nan时，返回值为nan。

## 约束说明

-   本接口不支持Subnormal场景：处于Subnormal范围内的输入和输出值，都会被刷新为保留符号的0。
-   本接口不受编译选项[--cce-use-fast-math](../../../../guides/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.md#常用的编译选项)的影响。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(256) void compute_fdivide_intrinsic(float *result, const float *x, const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = __fdividef(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_fdivide_intrinsic_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = __fdividef(x[idx], y[idx]);
    }

    __global__ __vector__ void run_fdivide_intrinsic(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        asc_vf_call<compute_fdivide_intrinsic_vf>(dim3(256), result, x, y, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
y: 1.5, 2.5, 3.5, 4.5
result: 0.1666667, 0.3, 0.3571429, 0.3888889
```
