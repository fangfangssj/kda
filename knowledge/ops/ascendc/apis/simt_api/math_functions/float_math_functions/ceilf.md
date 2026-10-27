---
type: API
title: Ceilf
description: 获取大于或等于输入数据的最小整数值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/ceilf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# ceilf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取大于或等于输入数据的最小整数值。

## 函数原型

```cpp
inline float ceilf(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

大于或等于输入数据的最小整数值。特别场景说明如下：

-   当`x`为nan时，返回值为nan。
-   当`x`为inf时，返回值为inf。
-   当`x`为-inf时，返回值为-inf。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_ceilf(float *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = ceilf(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_ceilf_vf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = ceilf(x[idx]);
    }

    __global__ __vector__ void run_ceilf(__gm__ float *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_ceilf_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 1, 1, 2, 2
```

- [Resize 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/resize_950.md)
- [UpsampleLinear1dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_linear_1d_backward_950.md)
- [UpsampleNearest2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_nearest_2d_backward_950.md)
- [UpsampleNearestExact2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_nearest_exact_2d_backward_950.md)
- [UpsampleNearestExact3dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_nearest_exact_3d_backward_950.md)
- [RoiPoolingWithArgMax 950 ACLNN 算子知识](../../../../operators/aclnn/cv/objdetect/roi_pooling_with_arg_max_950.md)
- [ResizeBilinear 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/resize_bilinear_950.md)
