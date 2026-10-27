---
type: API
title: __hfma_relu
description: 对输入half类型数据x、y、z，计算x与y相乘加上z的结果，并遵循CAST_RINT模式对结果进行舍入。负数结果置为0。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_arithmetic_functions/__hfma_relu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hfma\_relu

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

对输入half类型数据x、y、z，计算x与y相乘加上z的结果，并遵循CAST\_RINT模式对结果进行舍入。负数结果置为0。

## 函数原型

```cpp
half __hfma_relu(const half x, const half y, const half z)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |
| z | 输入 | 源操作数。 |

## 返回值说明

x \* y + z的值，若结果为负数，置为0。本接口受全局饱和模式影响，特殊值如下：

<table>
  <tr>
    <th>x值</th>
    <th>y值</th>
    <th>z值</th>
    <th>非饱和模式返回值</th>
    <th>饱和模式返回值</th>
  </tr>
  <tr>
    <td>±inf</td>
    <td>±0</td>
    <td>—</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td>±0</td>
    <td>±inf</td>
    <td>—</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td colspan="2">x*y = inf</td>
    <td>-inf</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td colspan="2">x*y = -inf</td>
    <td>inf</td>
    <td>nan</td>
    <td>0</td>
  </tr>
  <tr>
    <td colspan="3">x*y+z超出ASCRT_MAX_NORMAL_FP16</td>
    <td>inf</td>
    <td>ASCRT_MAX_NORMAL_FP16</td>
  </tr>
  <tr>
    <td colspan="3">x*y+z小于ASCRT_MIN_NORMAL_FP16</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td colspan="3">x、y、z任意一个为nan</td>
    <td>nan</td>
    <td>0</td>
  </tr>
</table>

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void KernelHfma_relu(half* dst, half* x, half* y, half* z)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hfma_relu(x[idx], y[idx], z[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHfma_relu(__gm__ half* dst, __gm__ half* x, __gm__ half* y, __gm__ half* z)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hfma_relu(x[idx], y[idx], z[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
y: -0.5, 0.5, 1.25, 2.0
z: 1.0, -1.0, 0.5, 2.5
result: 1.625, 0, 3.625, 0
```
