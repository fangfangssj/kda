---
type: API
title: __hbltx2
description: 比较两个bfloat16x2_t类型数据的两个分量，仅当两个分量均满足第一个数小于第二个数时，返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16x2_compare_functions/__hbltx2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hbltx2

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

比较两个bfloat16x2\_t类型数据的两个分量，仅当两个分量均满足第一个数小于第二个数时，返回true。

## 函数原型

```cpp
bool __hbltx2(bfloat16x2_t x, bfloat16x2_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

比较输入各分量是否均满足第一个数小于第二个数的结果。

-   true：各分量均满足第一个数小于第二个数。
-   false：任一分量不满足第一个数小于第二个数。

特殊值如下：

<table>
  <thead>
    <tr>
      <th>x分量值</th>
      <th>y分量值</th>
      <th>返回值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">任一分量含nan</td>
      <td>false</td>
    </tr>
    <tr>
      <td>inf</td>
      <td>正常值</td>
      <td>false</td>
    </tr>
    <tr>
      <td>正常值</td>
      <td>inf</td>
      <td>true</td>
    </tr>
    <tr>
      <td>-inf</td>
      <td>正常值</td>
      <td>true</td>
    </tr>
    <tr>
      <td>±0</td>
      <td>±0</td>
      <td>false</td>
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
    __global__ __launch_bounds__(1024) void simt_hbltx2(bfloat16_t* x, bfloat16_t* y, bool* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        bfloat16x2_t* input1 = (bfloat16x2_t*)x;
        bfloat16x2_t* input2 = (bfloat16x2_t*)y;
        dst[idx] = __hbltx2(input1[idx], input2[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_hbltx2(__gm__ bfloat16x2_t* x, __gm__ bfloat16x2_t* y, __gm__ bool* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __hbltx2(x[idx], y[idx]);
    }

    __global__ __vector__ void compare_kernel(__gm__ bfloat16_t* x, __gm__ bfloat16_t* y, __gm__ bool* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_hbltx2>(dim3(1024), (__gm__ bfloat16x2_t*)x, (__gm__ bfloat16x2_t*)y, dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1.0, 1.0), (-1.0, 1.0), (-inf, inf), (nan, 0.5)
y: (0.0, 2.0), (0.0, 1.0), (inf, inf), (0.5, nan)
result: true, false, false, false
```
