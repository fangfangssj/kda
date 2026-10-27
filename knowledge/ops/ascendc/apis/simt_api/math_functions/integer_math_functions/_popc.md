---
type: API
title: __popc（整数数学函数）
description: 统计输入数据从二进制的高位到低位比特位为1的数量。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__popc.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_popc

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

统计输入数据从二进制的高位到低位比特位为1的数量。

## 函数原型

```cpp
int __popc(unsigned int x)
```

```cpp
int __popc(unsigned long long x)
```

```cpp
int __popc(unsigned long x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入数据二进制中比特位为1的数量。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelPopc(int* dst, unsigned int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __popc(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelPopc(int* dst, unsigned long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __popc(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelPopc(__gm__ int* dst, __gm__ unsigned int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __popc(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelPopc(__gm__ int* dst, __gm__ unsigned long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __popc(x[idx]);
    }
    ```

输入输出示例如下：

- 当输入数据类型为`unsigned int`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 1, 1, 1, 4
    ```

- 当输入数据类型为`unsigned long long`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 1, 1, 1, 4
    ```
