---
type: API
title: __ull2bfloat16_rn
description: 遵循CAST_RINT模式，将uint64类型数据转换为bfloat16类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__ull2bfloat16_rn.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_ull2bfloat16\_rn

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_RINT模式，将uint64类型数据转换为bfloat16类型数据，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __ull2bfloat16_rn(const unsigned long long int x)
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
| 257 | 256 |
| 259 | 260 |
| 18446744073709551615（ULLONG_MAX） | 1.84467e+19 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__ull2bfloat16_rn(bfloat16_t* dst, uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2bfloat16_rn(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ull2bfloat16_rn(__gm__ bfloat16_t* dst, __gm__ uint64_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __ull2bfloat16_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 129, 257, 258, 259, 1000000000000, 9223372036854775808, 18446744073709551615, 0
result: 129.0, 256.0, 258.0, 260.0, 1000727379968.0, 9.22337e+18, 1.84467e+19, 0.0
```
