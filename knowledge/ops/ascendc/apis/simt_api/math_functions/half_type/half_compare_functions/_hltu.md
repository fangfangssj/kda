---
type: API
title: __hltu
description: 比较两个half类型数据，当第一个数小于第二个数时，返回true。若任一输入为nan，返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_compare_functions/__hltu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hltu

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

比较两个half类型数据，当第一个数小于第二个数时，返回true。若任一输入为nan，返回true。

## 函数原型

```cpp
bool __hltu(half x, half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

比较输入的第一个数是否小于第二个数的结果。任一输入为nan时返回true。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel_hltu(bool* dst, half* x, half* y, uint32_t total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hltu(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_hltu(__gm__ bool* dst, __gm__ half* x, __gm__ half* y, uint32_t total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hltu(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, nan, 2.5
y: 0.75, -0.25, 3.25, nan
result: true, false, true, true
```
