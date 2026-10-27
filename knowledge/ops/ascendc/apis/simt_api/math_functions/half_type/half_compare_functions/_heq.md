---
type: API
title: __heq
description: 比较两个half类型数据是否相等，相等时返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_compare_functions/__heq.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_heq

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

比较两个half类型数据是否相等，相等时返回true。

## 函数原型

```cpp
bool __heq(half x, half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

比较输入数据是否相等的结果，饱和与非饱和模式行为一致。特殊值如下：

<table>
  <tr>
    <th>x值</th>
    <th>y值</th>
    <th>返回值</th>
  </tr>
  <tr>
    <td colspan="2">x=y，均为非nan的有限值</td>
    <td>true</td>
  </tr>
  <tr>
    <td colspan="2">x≠y，均为非nan的有限值</td>
    <td>false</td>
  </tr>
  <tr>
    <td>±0</td>
    <td>±0</td>
    <td>true</td>
  </tr>
  <tr>
    <td>inf</td>
    <td>inf</td>
    <td>true</td>
  </tr>
  <tr>
    <td>-inf</td>
    <td>-inf</td>
    <td>true</td>
  </tr>
  <tr>
    <td>inf</td>
    <td>-inf</td>
    <td>false</td>
  </tr>
  <tr>
    <td colspan="2">任一操作数为nan</td>
    <td>false</td>
  </tr>
</table>

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void KernelHeq(bool* dst, half* x, half* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __heq(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHeq(__gm__ bool* dst, __gm__ half* x, __gm__ half* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __heq(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
y: -1.25, -0.25, 3.25, nan
result: true, false, false, false
```
