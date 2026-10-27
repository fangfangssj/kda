---
type: API
title: __mul64hi
description: 获取输入int64类型数据x和y乘积的高64位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__mul64hi.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_mul64hi

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取输入int64类型数据x和y乘积的高64位。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002562010435.png)

## 函数原型

```cpp
long long __mul64hi(long long x, long long y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入int64类型数据x和y乘积的高64位。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelMul64hi(long long* dst, long long* x, long long* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul64hi(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelMul64hi(__gm__ long long* dst, __gm__ long long* x,  __gm__ long long* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mul64hi(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -12, 23, -7, 6
y: 6, 3, 4, 15
result: -1, 0, -1, 0
```
