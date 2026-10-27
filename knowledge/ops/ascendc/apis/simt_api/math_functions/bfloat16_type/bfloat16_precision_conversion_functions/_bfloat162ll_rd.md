---
type: API
title: __bfloat162ll_rd
description: 遵循CAST_FLOOR模式，将bfloat16类型数据转换为64位有符号整数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat162ll_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162ll\_rd

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_FLOOR模式，将bfloat16类型数据转换为64位有符号整数，返回转换后的值。

## 函数原型

```cpp
inline long long int __bfloat162ll_rd(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的64位有符号整数。特别场景说明如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | 0 |
| inf | 9223372036854775807（INT64_MAX） |
| ASCRT_MAX_NORMAL_BF16 | 9223372036854775807（INT64_MAX） |
| -inf | -9223372036854775808（INT64_MIN） |
| -ASCRT_MAX_NORMAL_BF16 | -9223372036854775808（INT64_MIN） |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__bfloat162ll_rd(int64_t* dst, bfloat16_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __bfloat162ll_rd(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__bfloat162ll_rd(__gm__ int64_t* dst, __gm__ bfloat16_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __bfloat162ll_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, 1.5, 2.5, -1.5, -2.5, 16777216.0, -16777216.0, inf
result: 0, 1, 2, -2, -3, 16777216, -16777216, 9223372036854775807
```
