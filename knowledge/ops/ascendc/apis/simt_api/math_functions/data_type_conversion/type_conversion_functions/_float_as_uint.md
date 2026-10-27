---
type: API
title: __float_as_uint
description: 将浮点数中的位重新解释为无符号整数，即将浮点数存储的位按照无符号整数的格式进行读取。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float_as_uint.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float\_as\_uint

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将浮点数中的位重新解释为无符号整数，即将浮点数存储的位按照无符号整数的格式进行读取。

## 函数原型

```cpp
inline unsigned int __float_as_uint(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入的浮点数中的位重新解释成的无符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0 | 0 |
| -0 | 2147483648 |
| nan | 2143289344 |
| inf | ASCRT_INF_U |
| -inf | 4286578688 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float_as_uint(uint32_t* dst, float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float_as_uint(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float_as_uint(__gm__ uint32_t* dst, __gm__ float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float_as_uint(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.0, -0.0, 1.0, -1.0, inf, -inf, nan
result: 0, 2147483648, 1065353216, 3212836864, 2139095040, 4286578688, 2143289344
```
