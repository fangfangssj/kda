---
type: API
title: __int2float_rz
description: 遵循CAST_TRUNC模式，将int32类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__int2float_rz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_int2float\_rz

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_TRUNC模式，将int32类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __int2float_rz(const int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式转换成的浮点数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 2147483647 (INT32_MAX)  | 2147483520 |
| -2147483648 (INT32_MIN) | -ASCRT_TWO_TO_31_F |
| 16777217 | ASCRT_TWO_TO_24_F |
| -16777217 | -ASCRT_TWO_TO_24_F |
| 33554433 | 33554432 |
| -33554433 | -33554432 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__int2float_rz(float* dst, int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int2float_rz(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__int2float_rz(__gm__ float* dst, __gm__ int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int2float_rz(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -16777213, -16777212, -5, 0, 5, 16777224, 16777225, 2147483643
result: -16777213.0, -16777212.0, -5.0, 0.0, 5.0, 16777224.0, 16777224.0, 2147483520.0
```
