---
type: API
title: Rnormf
description: 获取输入数据a中前n个元素的平方和a\[0\]^2 + a\[1\]^2 + ...+ a\[n-1\]^2的平方根的倒数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.math_function
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/rnormf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# rnormf

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取输入数据a中前n个元素的平方和a\[0\]^2 + a\[1\]^2 + ...+ a\[n-1\]^2的平方根的倒数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002516816371.png)

## 函数原型

```cpp
inline float rnormf(int n, float* a)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| n | 输入 | 源操作数。输入数据a中连续计算的元素个数。 |
| a | 输入 | 源操作数。UB、Global Memory或栈空间的地址。 |

## 返回值说明

a\[0\]^2 + a\[1\]^2 + ...+ a\[n-1\]^2的平方根的倒数。

-   若a\[0\]^2 + a\[1\]^2 + ...+ a\[n-1\]^2的平方根超出float最大范围，返回值为0。
-   若a\[0\]^2 + a\[1\]^2 + ...+ a\[n-1\]^2平方根的倒数超出float最大范围，返回值为inf。
-   若a\[0\]、a\[1\] 、... 、a\[n-1\]都为0，返回值为inf。
-   若a\[0\]、a\[1\] 、... 、a\[n-1\]任意一个或多个为±inf，返回值为0。
-   若a\[0\]、a\[1\] 、... 、a\[n-1\]任意一个或多个为nan同时不是±inf，返回值为nan。
-   若n小于1，返回a\[0\]绝对值的倒数，若a\[0\]为0，返回值为inf。

## 约束说明

-   输入数据a的长度必须大于等于参数n。
-   若n过大，接口性能无法保证。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(256) void compute_rnormf(float *result, const int *n, float *vector_data, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = rnormf(n[idx], vector_data + idx * 4);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(256) inline void compute_rnormf_vf(__gm__ float *result, __gm__ const int *n, __gm__ float *vector_data, uint32_t count)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        result[idx] = rnormf(n[idx], vector_data + idx * 4);
    }

    __global__ __vector__ void run_rnormf(__gm__ float *result, __gm__ const int *n, __gm__ float *vector_data, uint32_t count)
    {
        asc_vf_call<compute_rnormf_vf>(dim3(256), result, n, vector_data, count);
    }
    ```

输入输出示例如下：

```
n: 1, 2, 3, 1
vector_data: [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
result: 1, 0.2773501, 0.1414214, 0.25
```
