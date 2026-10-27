---
type: API
title: __e5m2x22float2
description: 将float8_e5m2x2_t类型数据的两个分量转换为浮点数，返回转换后的float2类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/fp8_type/fp8_conversion_functions/__e5m2x22float2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp8.h
---

# \_\_e5m2x22float2

## 功能说明

头文件路径为：`"simt_api/asc_fp8.h"`。

将float8_e5m2x2_t类型数据的两个分量转换为浮点数，返回转换后的float2类型数据。

## 函数原型

```cpp
inline float2 __e5m2x22float2(const float8_e5m2x2_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

将float8\_e5m2x2\_t类型输入转换成的float2类型数据。特殊值如下：

| x分量值 | 返回值 |
|---|---|
| 0 | 0 |
| -0 | -0 |
| nan | nan |
| inf | inf |
| -inf | -inf |

## 约束说明

SIMT编程场景当前不支持使用该接口。

## 调用示例

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp8.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_e5m2x22float2(__gm__ float8_e5m2x2_t* input, __gm__ float2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个float8_e5m2x2_t类型的数据，即2个fp8_e5m2_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        output[idx] = __e5m2x22float2(input[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ uint8_t* input,  __gm__ float* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_e5m2x22float2>(dim3(1024), (__gm__ float8_e5m2x2_t*)input, (__gm__ float2*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x(float8_e5m2x2_t)：(0, -0), (0.5, -0.5), (8, -8), (inf, nan)
result(float2)：(0, -0), (0.5, -0.5), (8, -8), (inf, nan)
```
