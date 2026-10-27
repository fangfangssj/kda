---
type: API
title: __hminx2
description: 获取两个half2类型数据各分量的最小值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half2_compare_functions/__hminx2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hminx2

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

获取两个half2类型数据各分量的最小值。

## 函数原型

```cpp
half2 __hminx2(const half2 x, const half2 y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据各分量的最小值。该接口受全局饱和寄存器影响，特殊值如下：

| x分量 | y分量 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- | --- |
| 0 | -0 | -0 | -0 |
| -0 | 0 | -0 | -0 |
| -0 | -0 | -0 | -0 |
| 0 | 0 | 0 | 0 |
| nan | 任意值 | nan | 0 |
| 任意值 | nan | nan | 0 |
| nan | nan | nan | 0 |
| 任意非nan值 | -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | 任意非nan值 | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |
| inf | inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_hminx2(half* x, half* y, half* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        half2* input1 = (half2*)x;
        half2* input2 = (half2*)y;
        half2* out = (half2*)dst;
        out[idx] = __hminx2(input1[idx], input2[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hminx2(__gm__ half2* x, __gm__ half2* y, __gm__ half2* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hminx2(x[idx], y[idx]);
    }

    __global__ __vector__ void compare_kernel(__gm__ half* x, __gm__ half* y, __gm__ half* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hminx2>(dim3(1024), (__gm__ half2*)x, (__gm__ half2*)y, (__gm__ half2*)dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (nan, 1.25), (0.75, nan)
y: (0.5, -2), (1.25, 3), (-2, nan), (nan, -0.25)
result: (-1.25, -2), (1.25, -3.75), (-2, 1.25), (0.75, -0.25)
```
