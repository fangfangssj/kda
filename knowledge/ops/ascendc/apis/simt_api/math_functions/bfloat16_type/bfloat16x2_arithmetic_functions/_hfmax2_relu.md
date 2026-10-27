---
type: API
title: __hfmax2_relu
description: 计算输入bfloat16x2_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST_RINT模式对结果进行舍入。负数结果置为0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/__hfmax2_relu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hfmax2\_relu

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

计算输入bfloat16x2\_t类型数据各分量的乘加的结果（前两个输入相乘后与第三个输入相加），并遵循CAST\_RINT模式对结果进行舍入。负数结果置为0。

## 函数原型

```cpp
bfloat16x2_t __hfmax2_relu(const bfloat16x2_t x, const bfloat16x2_t y, const bfloat16x2_t z)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |
| z | 输入 | 源操作数。 |

## 返回值说明

输入数据各分量乘加的结果。计算的分量a、b、c满足：

<table>
  <thead>
    <tr>
      <th>x分量值</th>
      <th>y分量值</th>
      <th>z分量值</th>
      <th>返回值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3">x、y、z任意一个为nan</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>inf</td>
      <td>0</td>
      <td>任意值</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>0</td>
      <td>inf</td>
      <td>任意值</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>inf</td>
      <td>inf</td>
      <td>-inf</td>
      <td>nan</td>
    </tr>
    <tr>
      <td>-inf</td>
      <td>-inf</td>
      <td>inf</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>inf</td>
      <td>正数</td>
      <td>非无穷大数值</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>-inf</td>
      <td>正数</td>
      <td>非无穷大数值</td>
      <td>0</td>
    </tr>
    <tr>
      <td>inf</td>
      <td>inf</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>-inf</td>
      <td>-inf</td>
      <td>0</td>
      <td>inf</td>
    </tr>
    <tr>
      <td colspan="3">x、y、z为正常值, 结果&lt;0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_hfmax2_relu(bfloat16_t* x, bfloat16_t* y, bfloat16_t* z, bfloat16_t* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        bfloat16x2_t* input1 = (bfloat16x2_t*)x;
        bfloat16x2_t* input2 = (bfloat16x2_t*)y;
        bfloat16x2_t* input3 = (bfloat16x2_t*)z;
        bfloat16x2_t* out = (bfloat16x2_t*)dst;
        out[idx] = __hfmax2_relu(input1[idx], input2[idx], input3[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hfmax2_relu(__gm__ bfloat16x2_t* x, __gm__ bfloat16x2_t* y, __gm__ bfloat16x2_t* z, __gm__ bfloat16x2_t* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hfmax2_relu(x[idx], y[idx], z[idx]);
    }

    __global__ __vector__ void compute_kernel(__gm__ bfloat16_t* x, __gm__ bfloat16_t* y, __gm__ bfloat16_t* z, __gm__ bfloat16_t* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hfmax2_relu>(dim3(1024), (__gm__ bfloat16x2_t*)x, (__gm__ bfloat16x2_t*)y, (__gm__ bfloat16x2_t*)z, (__gm__ bfloat16x2_t*)dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (-2, 1.25), (0.75, -0.25)
y: (0.5, -2), (1.25, 3), (1.5, -1), (-0.5, 2.25)
z: (-0.5, 1), (-2, 0.75), (1, -1.5), (0.25, 2)
result: (0, 0), (1.125, 0), (0, 0), (0, 1.4375)
```
