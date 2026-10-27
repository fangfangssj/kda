---
type: API
title: __ull2float_rn
description: 遵循CAST_RINT模式，将uint64类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/data_type_conversion/type_conversion_functions/__ull2float_rn.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_ull2float\_rn

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

遵循CAST\_RINT模式，将uint64类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __ull2float_rn(const unsigned long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换成的浮点数。特殊值如下：

| x值 | 返回值 |
|---|---|
| 16777217 (2^24+1) | 16777216.0 |
| 33554433 (2^25+1) | 33554432.0 |
| 4294967295 (UINT32_MAX) | 4294967296.0 |
| 18446744073709551615 (UINT64_MAX) | 1.84467441e+19 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void kernel__ull2float_rn(float* dst, uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2float_rn(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ull2float_rn(__gm__ float* dst, __gm__ uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2float_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 2, 3, 16777218, 16777220, 4294967294, 9007199254740992, 9223372036854775806, 18446744073709551614
result: 2.0, 3.0, 16777218.0, 16777220.0, 4294967296.0, 9007199254740992.0, 9.22337204e+18, 1.84467441e+19
```
