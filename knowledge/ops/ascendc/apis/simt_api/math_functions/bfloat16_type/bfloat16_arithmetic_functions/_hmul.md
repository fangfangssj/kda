---
type: API
title: __hmul
description: 计算两个bfloat16类型数据相乘的结果，并遵循CAST_RINT模式对结果进行舍入。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_arithmetic_functions/__hmul.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hmul

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

计算两个bfloat16类型数据相乘的结果，并遵循CAST\_RINT模式对结果进行舍入。

## 函数原型

```cpp
bfloat16_t __hmul(const bfloat16_t x, const bfloat16_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据相乘的结果。本接口不受全局饱和模式影响，特殊值如下：

-   当输入和结果都不为nan时，`x*y`的符号为`x`和`y`符号的异或。
-   `__hmul(x, y)`等价于`__hmul(y, x)`。
-   `x`为非0值，`y`为±inf时，返回值符号由`x`和`y`的符号异或决定，值为inf。
-   `x`为±0，`y`为±inf时，返回值为nan。
-   `x`为±0，`y`为有限值时，返回值符号由`x`和`y`的符号异或决定，值为0。
-   `x`、`y`任意一个为nan时，返回值为nan。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel_hmul(bfloat16_t* dst, bfloat16_t* x, bfloat16_t* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hmul(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_hmul(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hmul(x[idx], y[idx]);
    }

    __global__ __vector__ void compute_kernel(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y, uint32_t total_length)
    {
        asc_vf_call<kernel_hmul>(dim3(1024), dst, x, y, total_length);
    }
    ```

输入输出示例如下：

```text
x: 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 4.0
y: 2.0, 0.5, -1.0, -1.0, 0.5, 2.0, 4.0, 0.5
result: 1.0, -0.25, -1.0, 1.0, 1.0, -4.0, 0.0, 2.0
```
