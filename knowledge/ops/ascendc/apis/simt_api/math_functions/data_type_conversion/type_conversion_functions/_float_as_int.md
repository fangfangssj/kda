---
type: API
title: __float_as_int
description: 将浮点数中的位重新解释为有符号整数，即将浮点数存储的位按照有符号整数的格式进行读取。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__float_as_int.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_float\_as\_int

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

将浮点数中的位重新解释为有符号整数，即将浮点数存储的位按照有符号整数的格式进行读取。

## 函数原型

```cpp
inline int __float_as_int(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入的浮点数中的位重新解释成的有符号整数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0 | 0 |
| -0 | ASCRT_MIN_VAL_S |
| nan | 2143289344 |
| inf | ASCRT_INT32_INF_S |
| -inf | -8388608 |
| ASCRT_MAX_NORMAL_F | 2139095039 |
| -ASCRT_MAX_NORMAL_F | -8388609 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__float_as_int(int32_t* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float_as_int(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float_as_int(__gm__ int32_t* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float_as_int(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.0, -0.0, 1.0, -1.0, inf, -inf, nan
result: 0, -2147483648, 1065353216, -1082130432, 2139095040, -8388608, 2143289344
```
