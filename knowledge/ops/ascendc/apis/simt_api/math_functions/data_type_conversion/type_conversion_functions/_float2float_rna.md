---
type: API
title: __float2float_rna
description: 将浮点数四舍五入取整，并遵循CAST_ROUND模式。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2float_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2float\_rna

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将浮点数四舍五入取整，并遵循CAST\_ROUND模式。

## 函数原型

```cpp
inline float __float2float_rna(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_ROUND模式取整后的浮点数。特殊值如下：

| x值 | 返回值 |
|---|---|
| ±0 | 0 |
| nan | nan |
| inf | inf |
| -inf | -inf |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float2float_rna(float* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2float_rna(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2float_rna(__gm__ float* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2float_rna(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -4.5, -1.25, 1.25, 4.5, inf, -inf, nan
result: -5.0, -1.0, 1.0, 5.0, inf, -inf, nan
```
