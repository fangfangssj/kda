---
type: API
title: __uint2float_rd
description: 遵循CAST_FLOOR模式，将uint32类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__uint2float_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_uint2float\_rd

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_FLOOR模式，将uint32类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __uint2float_rd(const unsigned int x)
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
| 0 | 0 |
| 16777217 (2^24+1) | 16777216 |
| 16777219 (2^24+3) | 16777218 |
| 33554434 (2^25+2) | 33554432 |
| 4294967295 (UINT32_MAX) | 4294967040 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__uint2float_rd(float* dst, uint32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __uint2float_rd(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__uint2float_rd(__gm__ float* dst, __gm__ uint32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __uint2float_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 1, 16777216, 16777217, 16777219, 33554434, 2147483648, 4294967295
result: 0.0, 1.0, 16777216.0, 16777216.0, 16777218.0, 33554432.0, 2147483648.0, 4294967040.0
```
