---
type: API
title: __ll2float_rd
description: 遵循CAST_FLOOR模式，将int64类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__ll2float_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_ll2float\_rd

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_FLOOR模式，将int64类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __ll2float_rd(const long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的浮点数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 9223372036854775807（INT64_MAX） | 9.223371e+18（2^63减2^40） |
| -9223372036854775808（INT64_MIN） | -9.223372e+18 |
| 16777217（2^24+1 ） | ASCRT_TWO_TO_24_F |
| -16777217（-2^24-1） | -1.67772e+07 |
| 9007199254740993（2^53+1） | 9.0072e+15 |
| -9007199254740993（-2^53-1） | -9.0072e+15 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__ll2float_rd(float* dst, int64_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __ll2float_rd(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ll2float_rd(__gm__ float* dst, __gm__ int64_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __ll2float_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -9007199254740993, -16777217, -1, 0, 1, 16777217, 9007199254740993, 9223372036854775807
result: -9007200328482816.0, -16777218.0, -1.0, 0.0, 1.0, 16777216.0, 9007199254740992.0, 9.22337149e+18
```
