---
type: API
title: Sincospif
description: 获取输入数据与π相乘的三角函数正弦值和余弦值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/sincospif.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# sincospif

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据与π相乘的三角函数正弦值和余弦值。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002484776398.png)

## 函数原型

```cpp
inline void sincospif(float x, float *s, float *c)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| s | 输出 | UB、Global Memory或栈空间的地址，用于存储输入数据与π相乘的三角函数正弦值。 |
| c | 输出 | UB、Global Memory或栈空间的地址，用于存储输入数据与π相乘的三角函数余弦值。 |

## 返回值说明

- 当输入x为有限值且Abs\(x\)\>=2^24时，输出参数s为0.0，输出参数c为1.0。
- 当输入x为整数且|Abs\(x\)\<2^24时，输出参数s为0.0。
- 当输入x为整数且|Abs\(x\)\<2^24时，若x为偶数，输出参数c返回1.0；若x为奇数，输出参数c返回-1.0。
- 当输入x=n+0.5，n为偶数，输出参数s返回1.0，n为奇数，输出参数s返回-1.0。
- 当输入x=n+0.5，输出参数c返回0.0。
- 当输入x为inf、-inf或nan时，输出参数s和c均为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_sincospif(float *sin_result, float *cos_result, const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        sincospif(x[idx], sin_result + idx, cos_result + idx);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_sincospif_vf(__gm__ float *sin_result, __gm__ float *cos_result, __gm__ const float *x, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        sincospif(x[idx], sin_result + idx, cos_result + idx);
    }

    __global__ __vector__ void run_sincospif(__gm__ float *sin_result, __gm__ float *cos_result, __gm__ const float *x, uint32_t count)
    {
        asc_vf_call<compute_sincospif_vf>(dim3(256), sin_result, cos_result, x, count);
    }
    ```

输入输出示例如下：

```
x: 0.25, 0.75, 1.25, 1.75
sin_result: 0.7071068, 0.7071068, -0.7071069, -0.7071065
cos_result: 0.7071068, -0.7071068, -0.7071066, 0.707107
```
