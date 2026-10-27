---
type: API
title: __umul64hi
description: 获取输入uint64类型数据x和y乘积的高64位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__umul64hi.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_umul64hi

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取输入uint64类型数据`x`和`y`乘积的高64位。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002531050522.png)

## 函数原型

```cpp
unsigned long long __umul64hi(unsigned long long x, unsigned long long y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入uint64类型数据x和y乘积的高64位。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelUmul64hi(unsigned long long* dst, unsigned long long* x, unsigned long long* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umul64hi(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelUmul64hi(__gm__ unsigned long long* dst, __gm__ unsigned long long* x, __gm__ unsigned long long* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umul64hi(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 4294967296, 4294967296, 8589934592, 17179869184, 18446744073709551615, 3, 5, 0
y: 4294967296, 8589934592, 4294967296, 4294967296, 18446744073709551615, 3, 5, 0
result: 1, 2, 2, 4, 18446744073709551614, 0, 0, 0
```
