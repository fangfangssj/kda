---
type: API
title: __bfloat162ushort_rd
description: 遵循CAST_FLOOR模式，将bfloat16_t类型数据转换为unsigned short int类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat162ushort_rd.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162ushort\_rd

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_FLOOR模式，将bfloat16_t类型数据转换为unsigned short int类型数据，返回转换后的值。

## 函数原型

```cpp
inline unsigned short int __bfloat162ushort_rd(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_FLOOR模式转换成的unsigned short int类型数据。特殊值如下：

| x值 | 返回值 |
|---|---|
| 0 | 0 |
| nan | 0 |
| inf | 65535（USHRT_MAX） |
| -inf | 0 |
| 超出short上界 | 65535（USHRT_MAX） |
| 负数 | 0 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel___bfloat162ushort_rd(unsigned short int* dst, bfloat16_t* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __bfloat162ushort_rd(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___bfloat162ushort_rd(__gm__ unsigned short int* dst, __gm__ bfloat16_t* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __bfloat162ushort_rd(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, 1.5, 2.5, -1.5, -2.5, 127.5, -127.5, inf
result: 0, 1, 2, 0, 0, 127, 0, 65535
```
