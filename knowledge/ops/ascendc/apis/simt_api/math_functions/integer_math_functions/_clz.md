---
type: API
title: __clz（整数数学函数）
description: 从输入数据的二进制最高有效位开始，返回连续的前导零的位数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__clz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_clz

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

从输入数据的二进制最高有效位开始，返回连续的前导零的位数。

## 函数原型

```cpp
int __clz(int x)
```

```cpp
int __clz(long long x)
```

```cpp
int __clz(long x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

返回输入整数二进制表示中连续高位零的数量。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelClz(int* dst, int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __clz(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelClz(int* dst, long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __clz(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelClz(__gm__ int* dst, __gm__ int* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __clz(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelClz(__gm__ int* dst, __gm__ long long* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __clz(x[idx]);
    }
    ```

输入输出示例如下：

- 当输入数据类型为`int`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 32, 31, 30, 28, 28
    ```

- 当输入数据类型为`long long`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 64, 63, 62, 60, 60
    ```
