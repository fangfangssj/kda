---
type: API
title: __float2int_rd
description: 遵循CAST_FLOOR模式，将浮点数转换为有符号整数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2int_rd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2int\_rd

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_FLOOR模式，将浮点数转换为有符号整数，返回转换后的值。

## 函数原型

```cpp
inline int __float2int_rd(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的有符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | 0 |
| inf | 2147483647 |
| -inf | -2147483648 |
| 超出 int 范围的正数 | 2147483647 |
| 超出 int 范围的负数 | -2147483648 |
| 正小数 | 向下取整 |
| 负小数 | 向下取整 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float2int_rd(int32_t* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2int_rd(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2int_rd(__gm__ int32_t* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2int_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -2.5, -1.5, 0.5, 1.5, inf, -inf, nan
result: -3, -2, 0, 1, 2147483647, -2147483648, 0
```
