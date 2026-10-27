---
type: API
title: abs
description: 获取float类型输入数据的绝对值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/float_math_functions/abs.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# abs

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取float类型输入数据的绝对值。

## 函数原型

```cpp
float abs(float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入数据的绝对值。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| nan | nan |
| inf | inf |
| -inf | inf |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(1024) void kernel_abs(float* dst, float* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = abs(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_abs(__gm__ float* dst, __gm__ float* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = abs(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.0, 2.5, -3.75
result: 1.25, 0.0, 2.5, 3.75
```
