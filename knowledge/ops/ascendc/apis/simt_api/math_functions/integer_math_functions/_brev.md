---
type: API
title: __brev（整数数学函数）
description: 将输入数据的位序反转，返回反转后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__brev.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_brev

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将输入数据的位序反转，返回反转后的值。

## 函数原型

```cpp
unsigned int __brev(unsigned int x)
```

```cpp
unsigned long long __brev(unsigned long long x)
```

```cpp
unsigned long __brev(unsigned long x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入数据的位反转值。当输入的类型为uint32_t时，返回值的第n位对应输入数据的第31-n位；当输入的类型为uint64_t时，返回值的第n位对应输入数据的第63-n位。

- 当x为0时，类型为uint32\_t时，返回值为0。
- 当x为0时，类型为uint64\_t时，返回值为0。
- 当x为1时，类型为uint32\_t时，返回值为2147483648。
- 当x为1时，类型为uint64\_t时，返回值为9223372036854775808。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel_brev(unsigned int* dst, unsigned int* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __brev(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel_brev(unsigned long long* dst, unsigned long long* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __brev(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_brev(__gm__ unsigned int* dst, __gm__ unsigned int* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __brev(x[idx]);
    }
    ```

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_brev(__gm__ unsigned long long* dst, __gm__ unsigned long long* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __brev(x[idx]);
    }
    ```

输入输出示例如下：

- 当输入数据类型为`unsigned int`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 2147483648, 1073741824, 268435456, 4026531840
    ```

- 当输入数据类型为`unsigned long long`时：

    ```text
    x: 0, 1, 2, 8, 15
    result: 0, 9223372036854775808, 4611686018427387904, 1152921504606846976, 17293822569102704640
    ```
