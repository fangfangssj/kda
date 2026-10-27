---
type: API
title: __short2half_rn
description: 遵循CAST_RINT模式，将short int类型数据转换为half类型数据，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__short2half_rn.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_short2half\_rn

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_RINT模式，将short int类型数据转换为half类型数据，返回转换后的值。

## 函数原型

```cpp
inline half __short2half_rn(const short int x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换成的half类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| 2049 | 2048 |
| -2049 | -2048 |
| 32767（short int最大值） | 32768 |
| -32768（short int最小值） | -32768 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel___short2half_rn(half* dst, short int* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __short2half_rn(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___short2half_rn(__gm__ half* dst, __gm__ short int* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __short2half_rn(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -4096, -2049, 0, 2049, 4096
result: -4096, -2048, 0, 2048, 4096
```
