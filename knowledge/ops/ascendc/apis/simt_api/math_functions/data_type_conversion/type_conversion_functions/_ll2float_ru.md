---
type: API
title: __ll2float_ru
description: 遵循CAST_CEIL模式，将int64类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__ll2float_ru.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_ll2float\_ru

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_CEIL模式，将int64类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __ll2float_ru(const long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_CEIL模式转换成的浮点数。特殊值如下：

| 输入 | 返回值 |
|---|---|
| 9223372036854775807（INT64_MAX） | 9.22337204e+18 |
| -9223372036854775808（INT64_MIN） | -9.22337204e+18 |
| 16777217（2^24+1 ） | 16777218 |
| -16777217（-2^24-1） | -ASCRT_TWO_TO_24_F |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__ll2float_ru(float* dst, int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2float_ru(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ll2float_ru(__gm__ float* dst, __gm__ int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2float_ru(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -9007199254740990, -16777220, -4, 0, 4, 16777220, 9007199254740990, 9223372036854775804
result: -9007198717870080.0, -16777220.0, -4.0, 0.0, 4.0, 16777220.0, 9007199254740992.0, 9.22337204e+18
```
