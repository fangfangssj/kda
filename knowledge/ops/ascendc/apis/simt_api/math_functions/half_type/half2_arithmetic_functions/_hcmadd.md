---
type: API
title: __hcmadd
description: 将三个half2输入视为复数，第一个分量为实部，第二个分量为虚部，执行复数乘加运算x*y+z。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half2_arithmetic_functions/__hcmadd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hcmadd

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将三个half2输入视为复数，第一个分量为实部，第二个分量为虚部，执行复数乘加运算x\*y+z。

## 函数原型

```cpp
half2 __hcmadd(const half2 x, const half2 y, const half2 z)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |
| z | 输入 | 源操作数。 |

## 返回值说明

输入数据视为复数，执行复数乘加运算的结果。对于输入x、y、z的结果如下：

| 实部结果 | 虚部结果 |
| --- | --- |
| __hfma(-x.y, y.y, __hfma(x.x, y.x, z.x)) | __hfma(x.y, y.x, __hfma(x.x, y.y, z.y)) |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_hcmadd(half* x, half* y, half* z, half* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        half2* input1 = (half2*)x;
        half2* input2 = (half2*)y;
        half2* input3 = (half2*)z;
        half2* out = (half2*)dst;
        out[idx] = __hcmadd(input1[idx], input2[idx], input3[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hcmadd(__gm__ half2* x, __gm__ half2* y, __gm__ half2* z, __gm__ half2* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hcmadd(x[idx], y[idx], z[idx]);
    }

    __global__ __vector__ void compute_kernel(__gm__ half* x, __gm__ half* y, __gm__ half* z, __gm__ half* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hcmadd>(dim3(1024), (__gm__ half2*)x, (__gm__ half2*)y, (__gm__ half2*)z, (__gm__ half2*)dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (-2, 1.25), (0.75, -0.25)
y: (0.5, -2), (1.25, 3), (1.5, -1), (-0.5, 2.25)
z: (-0.5, 1), (-2, 0.75), (1, -1.5), (0.25, 2)
result: (-0.125, 3.75), (12.375, 3.5625), (-0.75, 2.375), (0.4375, 3.8125)
```
