---
type: API
title: __halves2half2
description: 将输入的数据分别填充为half2前后两个分量，返回填充后的数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__halves2half2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_halves2half2

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将输入的数据分别填充为half2前后两个分量，返回填充后的数据。

## 函数原型

```cpp
inline half2 __halves2half2(const half x, const half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入的数据分别填充为half2前后两个分量的结果。特殊值如下：

| x（y）值 | 返回值 |
| --- | --- |
| 0 | 0 |
| -0 | -0 |
| nan | nan |
| inf | inf |
| -inf | -inf |
| ASCRT\_MAX\_NORMAL\_FP16 | ASCRT\_MAX\_NORMAL\_FP16 |
| -ASCRT\_MAX\_NORMAL\_FP16 | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __aicore__ void simt_halves2half2(half* input1, half* input2, half2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __halves2half2(input1[idx], input2[idx]);
    }
    __global__ __launch_bounds__(1024) void cast_kernel(half* input1,  half* input2, half* output, uint32_t input_total_length)
    {
        simt_halves2half2(input1, input2, (half2*)output, input_total_length);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_halves2half2(__gm__ half* input1, __gm__ half* input2, __gm__ half2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __halves2half2(input1[idx], input2[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ half* input1,  __gm__ half* input2, __gm__ half* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_halves2half2>(dim3(1024), input1, input2, (__gm__ half2*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
y: -0.5, 0.5, 1.25, 2.0
result: (-1.25, -0.5), (0.5, 0.5), (2.5, 1.25), (-3.75, 2.0)
```
