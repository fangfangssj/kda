---
type: API
title: __float2ll_rna
description: 遵循CAST_ROUND模式，将浮点数转换为64位有符号整数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2ll_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2ll\_rna

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_ROUND模式，将浮点数转换为64位有符号整数，返回转换后的值。

## 函数原型

```cpp
inline long long int __float2ll_rna(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_ROUND模式转换成的64位有符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | 0 |
| inf | 9223372036854775807（INT64_MAX） |
| -inf | -9223372036854775808（INT64_MIN） |
| 超出int64范围的正值（如ASCRT_MAX_NORMAL_F） | 9223372036854775807（INT64_MAX） |
| 超出int64范围的负值（如-ASCRT_MAX_NORMAL_F） | -9223372036854775808（INT64_MIN） |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float2ll_rna(int64_t* dst, float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2ll_rna(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2ll_rna(__gm__ int64_t* dst, __gm__ float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2ll_rna(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -4.5, -1.25, 1.25, 4.5, inf, -inf, nan
result: -5, -1, 1, 5, 9223372036854775807, -9223372036854775808, 0
```
