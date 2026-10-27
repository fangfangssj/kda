---
type: API
title: __int_as_float
description: 将整数中的位重新解释为浮点数，即将整数存储的位按照float的格式进行读取。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__int_as_float.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_int\_as\_float

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将整数中的位重新解释为浮点数，即将整数存储的位按照float的格式进行读取。

## 函数原型

```cpp
inline float __int_as_float(const int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入整数中的位重新解释成的浮点数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0x00000000 | 0 |
| 0x80000000 | -0 |
| 0x7F800000 | inf |
| 0xFF800000 | -inf |
| 0x7FFFFFFF | nan |
| 0x7F7FFFFF | ASCRT_MAX_NORMAL_F |
| 0xFF7FFFFF | -ASCRT_MAX_NORMAL_F |
| 0x00000001 | ASCRT_MIN_DENORM_F |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__int_as_float(float* dst, int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int_as_float(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__int_as_float(__gm__ float* dst, __gm__ int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int_as_float(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, -2147483648, 1, 1065353216, -1082130432, 2139095040, -8388608, 2143289344
result: 0.0, -0.0, 1e-45, 1.0, -1.0, inf, -inf, nan
```
