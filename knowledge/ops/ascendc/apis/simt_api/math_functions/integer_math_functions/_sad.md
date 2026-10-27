---
type: API
title: __sad（整数数学函数）
description: 对输入数据x、y、z，计算 x - y + z的结果，即前两个输入作差的绝对值与第三个输入的和。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__sad.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_sad

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

对输入数据`x`、`y`、`z`，计算`|x - y| + z`的结果，即前两个输入作差的绝对值与第三个输入的和。

## 函数原型

```cpp
unsigned int __sad(int x, int y, unsigned int z)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，int32_t类型。 |
| y | 输入 | 源操作数，int32_t类型。 |
| z | 输入 | 源操作数，uint32_t类型。 |

## 返回值说明

`|x - y| + z`的结果。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelSad(unsigned int* dst, int* x, int* y, unsigned int* z)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __sad(x[idx], y[idx], z[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelSad(__gm__ unsigned int* dst, __gm__ int* x, __gm__ int* y, __gm__ unsigned int* z)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __sad(x[idx], y[idx], z[idx]);
    }
    ```

输入输出示例如下：

```text
x: -12, 0, 23, 7
y: 0, 15, 7, -13
z: 0, 5, 9, 3
result: 12, 20, 25, 23
```
