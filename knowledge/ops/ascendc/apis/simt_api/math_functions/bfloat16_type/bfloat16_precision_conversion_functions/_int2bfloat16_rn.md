---
type: API
title: __int2bfloat16_rn
description: 遵循CAST_RINT模式，将int32类型数据转换为bfloat16类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__int2bfloat16_rn.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_int2bfloat16\_rn

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_RINT模式，将int32类型数据转换为bfloat16类型数据，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __int2bfloat16_rn(const int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换成的bfloat16类型数据。特殊值如下：

| 输入 | 返回值 |
|---|---|
| 0 | 0 |
| ±257 | ±256 |
| ±259 | ±260 |
| ±514 | ±512 |
| 2147483647（INT32_MAX） | 2.14748e+09 |
| -2147483648（INT32_MIN） | -2.14748e+09 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__int2bfloat16_rn(bfloat16_t* dst, int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int2bfloat16_rn(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__int2bfloat16_rn(__gm__ bfloat16_t* dst, __gm__ int32_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __int2bfloat16_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 129, 257, 258, 259, -129, -258, 1000000, -1000000
result: 129.0, 256.0, 258.0, 260.0, -129.0, -258.0, 999424.0, -999424.0
```
