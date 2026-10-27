---
type: API
title: __hadd（整数数学函数）
description: 获取int32类型数据x和y的平均值，避免中间求和溢出。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__hadd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_hadd

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取int32类型数据x和y的平均值，避免中间求和溢出。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002555143760.png)

## 函数原型

```cpp
int __hadd(int x, int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据的平均值。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelHadd(int* dst, int* x, int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hadd(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHadd(__gm__ int* dst, __gm__ int* x, __gm__ int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hadd(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -7, 0, 23, -13
y: 6, 0, 15, -8
result: -1, 0, 19, -11
```
