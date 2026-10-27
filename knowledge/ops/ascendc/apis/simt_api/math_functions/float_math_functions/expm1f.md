---
type: API
title: Expm1f
description: 指定输入x，获取e的x次方减1。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/expm1f.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# expm1f

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

指定输入x，获取e的x次方减1。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002484776378.png)

## 函数原型

```cpp
inline float expm1f(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

e的x次方减1。

-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为-1.0。
-   当x为nan时，返回值为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_expm1f(float *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = expm1f(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_expm1f_vf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = expm1f(x[idx]);
    }

    __global__ __vector__ void run_expm1f(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_expm1f_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 0.2840254, 1.117, 2.490343, 4.754602
```

- [Expm1 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/expm1_950.md)
- [InplaceExpm1 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/inplace_expm1_950.md)
- [ForeachExpm1 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_expm1_950.md)
