---
type: API
title: __floats2half2_rn
description: 将输入数据x、y遵循CAST_RINT模式分别转换为half类型，并填充到half2的前后两部分，返回转换后的half2类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__floats2half2_rn.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_floats2half2\_rn

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将输入数据x、y遵循CAST\_RINT模式分别转换为half类型，并填充到half2的前后两部分，返回转换后的half2类型数据。

## 函数原型

```cpp
inline half2 __floats2half2_rn(const float x, const float y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

将输入float类型数据遵循CAST\_RINT模式分别转换为half类型，并填充到half2的前后两部分的结果。本接口受全局饱和寄存器影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| -0 | -0 | -0 |
| nan | nan | 0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| x>ASCRT\_MAX\_NORMAL\_FP16 | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |
| x<-ASCRT\_MAX\_NORMAL\_FP16 | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __aicore__ void simt_floats2half2_rn(float* input1, float* input2, half2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __floats2half2_rn(input1[idx], input2[idx]);
    }
    __global__ __launch_bounds__(1024) void cast_kernel(float* input1, float* input2, half* output, uint32_t input_total_length)
    {
        simt_floats2half2_rn(input1, input2, (half2*)output, input_total_length);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_floats2half2_rn(__gm__ float* input1, __gm__ float* input2, __gm__ half2* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > input_total_length) {
            return;
        }
        output[idx] = __floats2half2_rn(input1[idx], input2[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ float* input1,  __gm__ float* input2, __gm__ half* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_floats2half2_rn>(dim3(1024), input1, input2, (__gm__ half2*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: 128, 128.0625, 128.1875, inf
y: -128.1875, 0.5, 2.5, nan
result: (128, -128.25), (128, 0.5), (128.25, 2.5), (inf, nan)
```
