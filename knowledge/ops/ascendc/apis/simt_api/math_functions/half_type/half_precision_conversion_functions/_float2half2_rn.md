---
type: API
title: __float2half2_rn
description: 遵循CAST_RINT模式，将float类型数据转换为half类型后复制到half2的两个分量，返回转换后的half2类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__float2half2_rn.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_float2half2\_rn

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_RINT模式，将float类型数据转换为half类型后复制到half2的两个分量，返回转换后的half2类型数据。

## 函数原型

```cpp
inline half2 __float2half2_rn(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换并复制到两个分量后的half2类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| ±0 | 两个分量均为±0 |
| nan | 两个分量均为nan |
| inf | 两个分量均为inf |
| -inf | 两个分量均为-inf|

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel___float2half2_rn(half2* dst, float* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __float2half2_rn(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___float2half2_rn(__gm__ half2* dst, __gm__ float* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __float2half2_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 128, 128.0625, 128.1875, -128.1875, inf, -inf, nan
result: (128, 128), (128, 128), (128.25, 128.25), (-128.25, -128.25), (inf, inf), (-inf, -inf), (nan, nan)
```
