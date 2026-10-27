---
type: API
title: __half2short_rd
description: 遵循CAST_FLOOR模式，将half类型数据转换为short int类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half2short_rd.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half2short\_rd

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_FLOOR模式，将half类型数据转换为short int类型数据，返回转换后的值。

## 函数原型

```cpp
inline short int __half2short_rd(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的short int类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| nan | 0 |
| inf | 32767（SHRT_MAX） |
| -inf | -32768（SHRT_MIN） |
| 超出short上界 | 32767（SHRT_MAX） |
| 超出short下界 | -32768（SHRT_MIN） |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel___half2short_rd(short int* dst, half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half2short_rd(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___half2short_rd(__gm__ short int* dst, __gm__ half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half2short_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -32768, -2048, -1.5, 0.5, 1.5, 2048, inf, nan
result: -32768, -2048, -2, 0, 1, 2048, 32767, 0
```
