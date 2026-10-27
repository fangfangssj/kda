---
type: API
title: __ushort_as_bfloat16
description: 将unsigned short int的数据按位重新解释为bfloat16，即将unsigned short int的数据存储的位按照bfloat16的格式进行读取。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__ushort_as_bfloat16.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_ushort\_as\_bfloat16

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

将unsigned short int的数据按位重新解释为bfloat16，即将unsigned short int的数据存储的位按照bfloat16的格式进行读取。

## 函数原型

```cpp
inline bfloat16_t __ushort_as_bfloat16(const unsigned short int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

unsigned short int的数据按位重新解释为bfloat16的值。特殊值如下：

| x值                            | 返回值  |
|-------------------------------|------|
| 0                             | 0    |
| 32640(0x7F80)                 | inf  |
| 65408(0xFF80)                 | -inf |
| 32641(0x7F81) ~ 32767(0x7FFF) | nan  |
| 65409(0xFF81) ~ 65535(0xFFFF) | nan  |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__ushort_as_bfloat16(bfloat16_t* dst, unsigned short int* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __ushort_as_bfloat16(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__ushort_as_bfloat16(__gm__ bfloat16_t* dst, __gm__ unsigned short int* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __ushort_as_bfloat16(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 32768, 16256, 48896, 1, 32640, 64512, 32256
result: 0.0, -0.0, 1.0, -0.5, 0.0, inf, -2.65846e+36, 4.25353e+37
```
