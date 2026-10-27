---
type: API
title: __bfloat1622float2
description: 将bfloat16x2的两个分量分别转换为float，并填充到float2后返回。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat1622float2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat1622float2

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

将bfloat16x2的两个分量分别转换为float，并填充到float2后返回。

## 函数原型

```cpp
inline float2 __bfloat1622float2(const bfloat16x2_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

将bfloat16x2的两个分量分别转换为float，并填充到float2的结果。特殊值如下：

| x分量值 | 对应分量返回值 |
|---|---|
| ±0 | ±0 |
| nan | nan |
| inf | inf |
| -inf | -inf |
| ASCRT_MAX_NORMAL_BF16 | 3.38953e+38 |
| -ASCRT_MAX_NORMAL_BF16 | -3.38953e+38 |
| ASCRT_MIN_DENORM_BF16 | 9.18355e-41 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __aicore__ void simt_bfloat1622float2(bfloat16x2_t* input, float2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx > input_total_length / 2) {
            return;
        }
        output[idx] = __bfloat1622float2(input[idx]);
    }
    __global__ __launch_bounds__(1024) void cast_kernel(bfloat16_t* input, float* output, uint32_t input_total_length)
    {
        simt_bfloat1622float2((bfloat16x2_t*)input, (float2*)output, input_total_length);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_bfloat1622float2(__gm__ bfloat16x2_t* input, __gm__ float2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx > input_total_length / 2) {
            return;
        }
        output[idx] = __bfloat1622float2(input[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ bfloat16_t* input, __gm__ float* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_bfloat1622float2>(dim3(1024), (__gm__ bfloat16x2_t*)input, (__gm__ float2*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (0.5, -1.0), (1.0, 2.0), (3.0, 4.0), (0.0, 3.0), (nan, 1.0), (1.0, 2.0), (inf, -inf), (nan, nan)
result: (0.5, -1.0), (1.0, 2.0), (3.0, 4.0), (0.0, 3.0), (nan, 1.0), (1.0, 2.0), (inf, -inf), (nan, nan)
```
