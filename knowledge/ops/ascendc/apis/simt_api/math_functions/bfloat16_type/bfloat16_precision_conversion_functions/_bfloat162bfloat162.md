---
type: API
title: __bfloat162bfloat162
description: 将输入数据填充为bfloat16x2前后两个分量，返回转换后的bfloat16x2类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat162bfloat162.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162bfloat162

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

将输入数据填充为bfloat16x2前后两个分量，返回转换后的bfloat16x2类型数据。

## 函数原型

```cpp
inline bfloat16x2_t __bfloat162bfloat162(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入的数据填充为bfloat16x2的数据。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0 | (0, 0) |
| -0 | (-0, -0) |
| nan | (nan, nan) |
| inf | (inf, inf) |
| -inf | (-inf, -inf) |
| ASCRT_MAX_NORMAL_BF16 | (ASCRT_MAX_NORMAL_BF16, ASCRT_MAX_NORMAL_BF16) |
| -ASCRT_MAX_NORMAL_BF16 | (-ASCRT_MAX_NORMAL_BF16, -ASCRT_MAX_NORMAL_BF16) |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __aicore__ void simt_bfloat162bfloat162(bfloat16_t* input, bfloat16x2_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __bfloat162bfloat162(input[idx]);
    }
    __global__ __launch_bounds__(1024) void cast_kernel(bfloat16_t* input, bfloat16_t* output, uint32_t input_total_length)
    {
        simt_bfloat162bfloat162(input, (bfloat16x2_t*)output, input_total_length);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_bfloat162bfloat162(__gm__ bfloat16_t* input, __gm__ bfloat16x2_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __bfloat162bfloat162(input[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ bfloat16_t* input, __gm__ bfloat16_t* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_bfloat162bfloat162>(dim3(1024), input, (__gm__ bfloat16x2_t*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: 0.5, -0.5, -0.0, 0.0, inf, -inf, nan, 1.5
result: (0.5, 0.5), (-0.5, -0.5), (-0.0, -0.0), (0.0, 0.0), (inf, inf), (-inf, -inf), (nan, nan), (1.5, 1.5)
```
