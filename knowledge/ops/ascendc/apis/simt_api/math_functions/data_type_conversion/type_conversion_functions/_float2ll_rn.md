---
type: API
title: __float2ll_rn
description: 遵循CAST_RINT模式，将浮点数转换为64位有符号整数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2ll_rn.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2ll\_rn

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_RINT模式，将浮点数转换为64位有符号整数，返回转换后的值。

## 函数原型

```cpp
inline long long int __float2ll_rn(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换成的64位有符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | 0 |
| inf | 9223372036854775807（INT64_MAX） |
| -inf | -9223372036854775808（INT64_MIN） |
| 超出int64范围的正值 | 9223372036854775807（INT64_MAX） |
| 超出int64范围的负值 | -9223372036854775808（INT64_MIN） |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float2ll_rn(int64_t* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2ll_rn(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2ll_rn(__gm__ int64_t* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2ll_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -3.5, -0.5, 2.5, 3.5, inf, -inf, nan
result: -4, 0, 2, 4, 9223372036854775807, -9223372036854775808, 0
```
