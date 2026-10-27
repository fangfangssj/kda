---
type: API
title: 数学函数powf
description: 获取输入数据x的y次幂。 !
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/powf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# powf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据x的y次幂。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002516816365.png)

## 函数原型

```cpp
inline float powf(float x, float y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，幂计算的底数。 |
| y | 输入 | 源操作数，幂计算的指数。 |

## 返回值说明

x的y次幂的结果。

-   若x^y超出float最大范围，返回值为inf。
-   若x为±0，y小于0并且为奇数，返回值为±inf。
-   若x为±0，y小于0并且不为奇数，返回值为inf。
-   若x为±0，y大于0并且为奇数，返回值为±0。
-   若x为±0，y大于0并且不为奇数，返回值为0。
-   若x为-1，y为±inf，返回值为1。
-   若x为1，y为任意值（包括nan），返回值为1。
-   若y为±0，x为任意值（包括nan），返回值为1。
-   若x小于0，y不为整数，返回值为nan。
-   若|x|<1，y为-inf，返回值为inf。
-   若|x|\>1，y为-inf，返回值为0。
-   若|x|<1，y为inf，返回值为0。
-   若|x|\>1，y为inf，返回值为inf。
-   若x为-inf，y小于0并且为奇数，返回值为-0。
-   若x为-inf，y小于0并且不为奇数，返回值为0。
-   若x为-inf，y大于0并且为奇数，返回值为-inf。
-   若x为-inf，y大于0并且不为奇数，返回值为inf。
-   若x为inf，y小于0，返回值为0。
-   若x为inf，y大于0，返回值为inf。
-   在如下边界场景，返回值为nan。
    -   x为nan，y不为0。
    -   y为nan，x不为1。
    -   x、y均为nan

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_powf(float *result, const float *x, const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = powf(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_powf_vf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = powf(x[idx], y[idx]);
    }

    __global__ __vector__ void run_powf(__gm__ float *result, __gm__ const float *x, __gm__ const float *y, uint32_t count)
    {
        asc_vf_call<compute_powf_vf>(dim3(256), result, x, y, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
y: 1.25, 2.25, 3.25, 4.25
result: 0.176777, 0.523465, 2.065178, 10.787274
```

- [ForeachPowScalar 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_pow_scalar_950.md)
