---
type: API
title: Isnan
description: 判断浮点数是否为nan。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/isnan.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# isnan

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

判断浮点数是否为nan。

## 函数原型

```cpp
inline bool isnan(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

-   false：输入非nan。
-   true：输入为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_isnan(bool *result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = isnan(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_isnan_vf(__gm__ bool *result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = isnan(x[idx]);
    }

    __global__ __vector__ void run_isnan(__gm__ bool *result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_isnan_vf>(dim3(256), result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
result: 0, 0, 0, 0
```

- [UpsampleTrilinear3d 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_trilinear_3d_950.md)
- [ForeachMaximumList 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_maximum_list_950.md)
- [GridSampler3D 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/grid_sampler_3d_950.md)
- [ForeachMaximumScalar 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_maximum_scalar_950.md)
- [ForeachMaximumScalarList 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_maximum_scalar_list_950.md)
- [ForeachMinimumList 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_minimum_list_950.md)
- [ForeachMinimumScalar 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_minimum_scalar_950.md)
- [MaxPool2dWithIndices 950 ACLNN 算子知识](../../../../operators/aclnn/nn/pooling/max_pool2d_with_indices_950.md)
- [MaxPool3dWithArgmax 950 ACLNN 算子知识](../../../../operators/aclnn/nn/pooling/max_pool3d_with_argmax_950.md)
- [Hardtanh 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/hardtanh_950.md)
