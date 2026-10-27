---
type: API
title: __hbnex2
description: 比较两个half2类型数据的两个分量是否不相等，仅当两个分量均不相等时返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half2_compare_functions/__hbnex2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hbnex2

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

比较两个half2类型数据的两个分量是否不相等，仅当两个分量均不相等时返回true。

## 函数原型

```cpp
bool __hbnex2(half2 x, half2 y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

比较输入数据各分量是否均不相等的结果。任一输入为nan时返回false。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_hbnex2(half* x, half* y, bool* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        half2* input1 = (half2*)x;
        half2* input2 = (half2*)y;
        dst[idx] = __hbnex2(input1[idx], input2[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hbnex2(__gm__ half2* x, __gm__ half2* y, __gm__ bool* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hbnex2(x[idx], y[idx]);
    }

    __global__ __vector__ void compare_kernel(__gm__ half* x, __gm__ half* y, __gm__ bool* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hbnex2>(dim3(1024), (__gm__ half2*)x, (__gm__ half2*)y, dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1, 0), (1, -1), (inf, -inf), (nan, 0.5)
y: (1, 0), (1, -2), (inf, inf), (0.5, nan)
result: false, false, false, false
```
