---
type: API
title: __ffs（整数数学函数）
description: 从二进制输入数据的最低位开始，查找第一个值为1的比特位的位置，并返回该位置的索引，索引从1开始计数；如果二进制数据中没有1，则返回0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__ffs.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_ffs

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

从二进制输入数据的最低位开始，查找第一个值为1的比特位的位置，并返回该位置的索引，索引从1开始计数；如果二进制数据中没有1，则返回0。

## 函数原型

```cpp
int __ffs(int x)
```

```cpp
int __ffs(long long x)
```

```cpp
int __ffs(long x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

数据二进制低位开始的第一个比特位为1的位置。

-   当x=0时，返回0。
-   当x=1时，返回1。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelFfs(int* dst, int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ffs(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelFfs(int* dst, long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ffs(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelFfs(__gm__ int* dst, __gm__ int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ffs(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelFfs(__gm__ int* dst, __gm__ long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ffs(x[idx]);
    }
    ```

输入输出示例如下：

- 当输入数据类型为`int`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 1, 2, 4, 1
    ```

- 当输入数据类型为`long long`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 1, 2, 4, 1
    ```
