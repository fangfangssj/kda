---
type: API
title: __hsubx2
description: 计算两个half2类型数据各分量的相减结果，并遵循CAST_RINT模式对结果进行舍入。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half2_arithmetic_functions/__hsubx2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hsubx2

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

计算两个half2类型数据各分量的相减结果，并遵循CAST\_RINT模式对结果进行舍入。

## 函数原型

```cpp
half2 __hsubx2(const half2 x, const half2 y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据各分量相减的结果。本接口受全局饱和寄存器的影响，特殊值如下：

<table>
  <tr>
    <th>x分量</th>
    <th>y分量</th>
    <th>非饱和模式返回值</th>
    <th>饱和模式返回值</th>
  </tr>
  <tr>
    <td>inf</td>
    <td>有限值</td>
    <td>±inf</td>
    <td>±ASCRT_MAX_NORMAL_FP16</td>
  </tr>
  <tr>
    <td>有限值</td>
    <td>±inf</td>
    <td>∓inf</td>
    <td>∓ASCRT_MAX_NORMAL_FP16</td>
  </tr>
  <tr>
    <td>±inf</td>
    <td>±inf</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td>±inf</td>
    <td>∓inf</td>
    <td>±inf</td>
    <td>±ASCRT_MAX_NORMAL_FP16</td>
  </tr>
  <tr>
    <td>±0</td>
    <td>∓0</td>
    <td>±0</td>
    <td>±0</td>
  </tr>
  <tr>
    <td colspan="2">x=y</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>nan</td>
    <td>任意值</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td>任意值</td>
    <td>nan</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td>ASCRT_MAX_NORMAL_FP16</td>
    <td>-ASCRT_MAX_NORMAL_FP16</td>
    <td>inf</td>
    <td>ASCRT_MAX_NORMAL_FP16</td>
  </tr>
  <tr>
    <td>-ASCRT_MAX_NORMAL_FP16</td>
    <td>ASCRT_MAX_NORMAL_FP16</td>
    <td>-inf</td>
    <td>-ASCRT_MAX_NORMAL_FP16</td>
  </tr>
</table>

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_hsubx2(half* x, half* y, half* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        half2* input1 = (half2*)x;
        half2* input2 = (half2*)y;
        half2* out = (half2*)dst;
        out[idx] = __hsubx2(input1[idx], input2[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hsubx2(__gm__ half2* x, __gm__ half2* y, __gm__ half2* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hsubx2(x[idx], y[idx]);
    }

    __global__ __vector__ void compute_kernel(__gm__ half* x, __gm__ half* y, __gm__ half* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hsubx2>(dim3(1024), (__gm__ half2*)x, (__gm__ half2*)y, (__gm__ half2*)dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1, 0), (1, -1), (inf, -inf), (nan, 0.5)
y: (1, 0), (1, -2), (inf, inf), (0.5, nan)
result: (-2, 0), (0, 1), (nan, -inf), (nan, nan)
```
