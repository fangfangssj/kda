---
type: API
title: __float2int_rz
description: 遵循CAST_TRUNC模式，将浮点数转换为有符号整数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2int_rz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2int\_rz

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_TRUNC模式，将浮点数转换为有符号整数，返回转换后的值。

## 函数原型

```cpp
inline int __float2int_rz(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式转换成的有符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | 0 |
| inf | 2147483647 |
| -inf | -2147483648 |
| 超过INT_MAX范围的正值 | 2147483647 |
| 低于INT_MIN范围的负值 | -2147483648 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float2int_rz(int32_t* dst, float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2int_rz(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2int_rz(__gm__ int32_t* dst, __gm__ float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2int_rz(x[idx]);
    }

    __global__ __vector__ void run_kernel__float2int_rz(__gm__ int32_t* dst, __gm__ float* x, uint32_t total_length)
    {
        asc_vf_call<kernel__float2int_rz>(dim3(256), dst, x, total_length);
    }
    ```

输入输出示例如下：

```text
x: -6.5, -1.75, 1.75, 6.5, inf, -inf, nan
result: -6, -1, 1, 6, 2147483647, -2147483648, 0
```
