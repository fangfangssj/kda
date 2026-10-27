---
type: API
title: __umul24
description: 获取uint32类型数据x和y的低24位乘积的低32位运算结果。x和y的高8位数据被忽略。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__umul24.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_umul24

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取uint32类型数据x和y的低24位乘积的低32位运算结果。x和y的高8位数据被忽略。

## 函数原型

```cpp
unsigned int __umul24(unsigned int x, unsigned int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据低24位乘积的低32位结果。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelUmul24(unsigned int* dst, unsigned int* x, unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umul24(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelUmul24(__gm__ unsigned int* dst, __gm__ unsigned int* x, __gm__ unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umul24(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 12, 23, 70
y: 6, 15, 3, 4
result: 0, 180, 69, 280
```
