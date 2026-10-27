---
type: API
title: __half2half2
description: 将half类型数据复制到half2的两个分量，返回转换后的half2类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half2half2.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half2half2

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将half类型数据复制到half2的两个分量，返回转换后的half2类型数据。

## 函数原型

```cpp
inline half2 __half2half2(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

两个分量均为输入x的half2类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 两个分量均为0 |
| nan | 两个分量均为nan |
| inf | 两个分量均为inf |
| -inf | 两个分量均为-inf |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel___half2half2(half2* dst, half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half2half2(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___half2half2(__gm__ half2* dst, __gm__ half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half2half2(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -2, -0.5, 0.5, 2, nan
result: (-2, -2), (-0.5, -0.5), (0.5, 0.5), (2, 2), (nan, nan)
```
