---
type: API
title: __half_as_short
description: 将half类型数据按位重新解释为short int类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half_as_short.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half\_as\_short

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将half类型数据按位重新解释为short int类型数据。

## 函数原型

```cpp
inline short int __half_as_short(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入的half类型数据中的位重新解释成的short int类型数据。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| nan | 32256 |
| inf | 31744 |
| -inf | -1024 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel___half_as_short(short int* dst, half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half_as_short(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel___half_as_short(__gm__ short int* dst, __gm__ half* x, uint32_t input_total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= input_total_length) {
            return;
        }
        dst[idx] = __half_as_short(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.0, -0.0, 5.96046448e-08, 1.0, -1.0, inf, -inf, nan
result: 0, -32768, 1, 15360, -17408, 31744, -1024, 32256
```
