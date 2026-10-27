---
type: API
title: __bfloat162char_rz
description: 遵循CAST_TRUNC模式，将bfloat16_t类型数据转换为signed char类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat162char_rz.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162char\_rz

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_TRUNC模式，将bfloat16_t类型数据转换为signed char类型数据，返回转换后的值。

## 函数原型

```cpp
inline signed char __bfloat162char_rz(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式转换成的signed char类型数据。特殊值如下：

| x值       | 返回值 |
|----------|---|
| 0        | 0 |
| nan      | 0 |
| inf      | 127（SCHAR_MAX） |
| -inf     | -128（SCHAR_MIN） |
| 超出char上界 | 127（SCHAR_MAX） |
| 超出char下界 | -128（SCHAR_MIN） |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel___bfloat162char_rz(signed char* dst, bfloat16_t* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __bfloat162char_rz(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___bfloat162char_rz(__gm__ signed char* dst, __gm__ bfloat16_t* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __bfloat162char_rz(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, 1.5, 2.5, -1.5, -2.5, 127.5, -127.5, inf
result: 0, 1, 2, -1, -2, 127, -127, 127
```
