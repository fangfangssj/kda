---
type: API
title: __mulhi（整数数学函数）
description: 获取输入int32类型数据x和y乘积的高32位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__mulhi.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_mulhi

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取输入int32类型数据x和y乘积的高32位。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002562010439.png)

## 函数原型

```cpp
int __mulhi(int x, int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入int32类型数据x和y乘积的高32位。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelMulhi(int* dst, int* x, int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mulhi(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelMulhi(__gm__ int* dst, __gm__ int* x, __gm__ int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mulhi(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -12, 23, -7, 6
y: 6, 3, 4, 15
result: -1, 0, -1, 0
```
