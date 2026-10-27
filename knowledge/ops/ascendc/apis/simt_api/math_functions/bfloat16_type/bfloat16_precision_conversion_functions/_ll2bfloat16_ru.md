---
type: API
title: __ll2bfloat16_ru
description: 遵循CAST_CEIL模式，将int64类型数据转换为bfloat16类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__ll2bfloat16_ru.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_ll2bfloat16\_ru

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_CEIL模式，将int64类型数据转换为bfloat16类型数据，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __ll2bfloat16_ru(const long long int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_CEIL模式转换成的bfloat16类型数据。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0 | 0 |
| 257 | 258 |
| -257 | -256 |
| 9223372036854775807（INT64_MAX） | 9.22337e+18 |
| -9223372036854775808（INT64_MIN） | -9.22337e+18 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__ll2bfloat16_ru(bfloat16_t* dst, int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2bfloat16_ru(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ll2bfloat16_ru(__gm__ bfloat16_t* dst, __gm__ int64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ll2bfloat16_ru(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 129, 257, 258, 259, -129, -258, 9223372036854775807, -9223372036854775808
result: 129.0, 258.0, 258.0, 260.0, -129.0, -258.0, 9.22337e+18, -9.22337e+18
```
