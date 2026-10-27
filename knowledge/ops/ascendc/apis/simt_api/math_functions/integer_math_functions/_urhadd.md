---
type: API
title: __urhadd
description: 获取uint32类型数据x和y的向上取整平均值，避免中间求和溢出。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__urhadd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_urhadd

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取uint32类型数据x和y的向上取整平均值，避免中间求和溢出。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002585584765.png)

## 函数原型

```cpp
unsigned int __urhadd(unsigned int x, unsigned int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据的向上取整平均值。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelUrhadd(unsigned int* dst, unsigned int* x, unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __urhadd(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelUrhadd(__gm__ unsigned int* dst, __gm__ unsigned int* x, __gm__ unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __urhadd(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 12, 23, 70
y: 6, 0, 15, 7
result: 3, 6, 19, 39
```
