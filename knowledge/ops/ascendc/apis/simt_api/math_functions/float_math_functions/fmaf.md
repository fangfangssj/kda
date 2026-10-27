---
type: API
title: 数学函数fmaf
description: 对输入数据x、y、z，计算x与y相乘加上z的结果。 !
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/fmaf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# fmaf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

对输入数据x、y、z，计算x与y相乘加上z的结果。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002531284496.png)

## 函数原型

```cpp
inline float fmaf(float x, float y, float z)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |
| z | 输入 | 源操作数。 |

## 返回值说明

x \* y + z的值。

-   x为±inf，y为±0，返回nan。
-   x为±0，y为±inf，返回nan。
-   x\*y为inf，z为-inf，返回nan。
-   x\*y为-inf，z为inf，返回nan。
-   x\*y+z超出对应类型范围的最大值，返回inf。
-   x\*y+z小于对应类型范围的最小值，返回-inf。
-   x、y、z任意一个为nan，返回nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_fmaf(float *result, const float *x, const float *y, const float *z, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = fmaf(x[idx], y[idx], z[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_fmaf_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, __gm__ const float *z, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = fmaf(x[idx], y[idx], z[idx]);
    }

    __global__ __vector__ void run_fmaf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, __gm__ const float *z, uint32_t count)
    {
        asc_vf_call<compute_fmaf_vf>(dim3(256), result, x, y, z, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
y: 1.5, 2.5, 3.5, 4.5
z: -0.5, 0.5, 1.5, 2.5
result: -0.125, 2.375, 5.875, 10.375
```

- [UpsampleLinear1dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_linear_1d_backward_950.md)
- [UpsampleTrilinear3d 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_trilinear_3d_950.md)
- [Expm1 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/expm1_950.md)
- [GQA 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/attention/gqa_950.md)
- [InplaceExpm1 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/inplace_expm1_950.md)
- [GroupedMatmul 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/gmm/grouped_matmul_950.md)
- [ApplyRotaryPosEmb 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/posembedding/apply_rotary_pos_emb_950.md)
