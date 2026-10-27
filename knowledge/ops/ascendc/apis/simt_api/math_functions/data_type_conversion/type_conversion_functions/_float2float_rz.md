---
type: API
title: __float2float_rz
description: 将浮点数四舍五入取整，并遵循CAST_TRUNC模式。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float2float_rz.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float2float\_rz

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将浮点数向零取整，并遵循CAST\_TRUNC模式。

## 函数原型

```cpp
inline float __float2float_rz(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_TRUNC模式取整后的浮点数。特殊值如下：

| 输入 | 返回值 |
|---|---|
| ±0 | 0 |
| ±0.5 | 0 |
| 1.5 | 1.0 |
| -1.5 | -1.0 |
| nan | nan |
| inf | inf |
| -inf | -inf |
| ASCRT_MAX_NORMAL_F | ASCRT_MAX_NORMAL_F |
| -ASCRT_MAX_NORMAL_F | -ASCRT_MAX_NORMAL_F |
| ASCRT_MIN_DENORM_F | 0 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

```cpp
#include "simt_api/device_functions.h"

__global__ __launch_bounds__(1024) void kernel__float2float_rz(float* dst, float* x, uint32_t total_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_length) {
        return;
    }
    dst[idx] = __float2float_rz(x[idx]);
}
```

- SIMD与SIMT混合编程场景：

```cpp
#include "simt_api/device_functions.h"

__simt_vf__ __launch_bounds__(1024) inline void kernel__float2float_rz(__gm__ float* dst, __gm__ float* x, uint32_t total_length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_length) {
        return;
    }
    dst[idx] = __float2float_rz(x[idx]);
}
```

输入输出示例如下：

```text
x: -6.5, -1.75, 1.75, 6.5, inf, -inf, nan
result: -6.0, -1.0, 1.0, 6.0, inf, -inf, nan
```
