---
type: API
title: __hdiv
description: 计算两个half类型数据相除的结果，并遵循CAST_RINT模式对结果进行舍入处理。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_arithmetic_functions/__hdiv.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hdiv

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

计算两个half类型数据相除的结果，并遵循CAST\_RINT模式对结果进行舍入处理。

## 函数原型

```cpp
half __hdiv(const half x, const half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据相除的结果。

-   当输入和结果都不为nan时，x/y的符号为x和y符号的异或。
-   本接口受全局饱和模式影响，特殊值如下：

    <table>
      <tr>
        <th>x值</th>
        <th>y值</th>
        <th>非饱和模式返回值</th>
        <th>饱和模式返回值</th>
      </tr>
      <tr>
        <td>±0</td>
        <td>±0</td>
        <td>nan</td>
        <td>0</td>
      </tr>
      <tr>
        <td>±inf</td>
        <td>±inf</td>
        <td>nan</td>
        <td>0</td>
      </tr>
      <tr>
        <td>有限值</td>
        <td>±inf</td>
        <td>返回值符号由x和y的符号异或决定，值为0</td>
        <td>返回值符号由x和y的符号异或决定，值为0</td>
      </tr>
      <tr>
        <td>±inf</td>
        <td>有限值</td>
        <td>返回值符号由x和y的符号异或决定，值为inf</td>
        <td>返回值符号由x和y的符号异或决定，值为ASCRT_MAX_NORMAL_FP16</td>
      </tr>
      <tr>
        <td>不为0</td>
        <td>±0</td>
        <td>返回值符号由x和y的符号异或决定，值为inf</td>
        <td>返回值符号由x和y的符号异或决定，值为ASCRT_MAX_NORMAL_FP16</td>
      </tr>
      <tr>
        <td>±0</td>
        <td>不为0</td>
        <td>返回值符号由x和y的符号异或决定，值为0</td>
        <td>返回值符号由x和y的符号异或决定，值为0</td>
      </tr>
      <tr>
        <td colspan="2">x，y任意一个为nan</td>
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

    __global__ __launch_bounds__(1024) void KernelHdiv(half* dst, half* x, half* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hdiv(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHdiv(__gm__ half* dst, __gm__ half* x, __gm__ half* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hdiv(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
y: 0.5, -2.0, 5.0, -0.25
result: -2.5, -0.25, 0.5, 15
```
