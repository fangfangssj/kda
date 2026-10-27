---
type: API
title: __float2bfloat16_rd
description: 遵循CAST_FLOOR模式，将浮点数转换为bfloat16精度，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__float2bfloat16_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_float2bfloat16\_rd

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_FLOOR模式，将浮点数转换为bfloat16精度，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __float2bfloat16_rd(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的bfloat16类型数据。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
|---|---|---|
| +0 | 0 | 0 |
| -0 | -0 | -0 |
| nan | nan | 0 |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | -inf | -ASCRT_MAX_NORMAL_BF16 |
| ASCRT_MAX_NORMAL_F | ASCRT_MAX_NORMAL_BF16 | ASCRT_MAX_NORMAL_BF16 |
| -ASCRT_MAX_NORMAL_F | -inf | -ASCRT_MAX_NORMAL_BF16 |
| 1.00390625 | 1.0 | 1.0 |
| -1.00390625 | -1.0078125 | -1.0078125 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__float2bfloat16_rd(bfloat16_t* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2bfloat16_rd(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2bfloat16_rd(__gm__ bfloat16_t* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2bfloat16_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 128.25, 128.5, 128.75, -128.5, 0.0, -0.0, inf, nan
result: 128.0, 128.0, 128.0, -129.0, 0.0, -0.0, inf, nan
```
