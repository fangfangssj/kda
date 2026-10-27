---
type: API
title: 数学函数jnf
description: 获取输入数据x的n阶第一类贝塞尔函数jn的值。 !
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/jnf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# jnf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据x的n阶第一类贝塞尔函数jn的值。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002484776432.png)

## 函数原型

```cpp
inline float jnf(int n, float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| n | 输入 | 源操作数。 |
| x | 输入 | 源操作数。 |

## 返回值说明

输入数据的n阶第一类贝塞尔函数jn的值。

-   当n<0时，返回值为nan。
-   当x=inf或-inf时，返回值为0。
-   当x=nan时，返回值为nan。
-   在快速计算模式（[--cce-use-fast-math](../../../../guides/programming_guide/compilation_and_execution/operator_compilation/ai_core_operator_compilation.md#常用的编译选项)设置为`true`）下，若内部计算的expf结果为Subnormal数据，则接口返回值为0。

## 约束说明

-  n的最大取值为128。
-  本接口受编译选项`--cce-use-fast-math`的影响。
    -   --cce-use-fast-math=true：开启快速计算模式。本接口内部实现使用到了expf，由于快速计算模式下expf不支持Subnormal场景，因此本接口不支持内部计算的expf结果为Subnormal数据的场景。
    -   --cce-use-fast-math=false：支持Subnormal场景的高精度计算。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_jnf(float *result, const int *n, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = jnf(n[idx], x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_jnf_vf(__gm__ float *result, __gm__ const int *n, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = jnf(n[idx], x[idx]);
    }

    __global__ __vector__ void run_jnf(__gm__ float *result, __gm__ const int *n, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_jnf_vf>(dim3(256), result, n, x, count);
    }
    ```

输入输出示例如下：

```
n: 1, 2, 3, 1
x: 0.25, 0.75, 1.25, 1.75
result: 0.124026, 0.06707399, 0.03686836, 0.5801562
```
