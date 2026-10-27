---
type: API
title: __hmin
description: 获取两个输入数据中的最小值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_compare_functions/__hmin.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hmin

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

获取两个输入数据中的最小值。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002501694974.png)

## 函数原型

```cpp
inline half __hmin(half x, half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据中的最小值。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | y值 | 返回值 |
| --- | --- | --- |
| ±0 | ±0 | 0 |
| nan | normal | normal |
| nan | nan | nan |
| -inf | normal | -inf |
| normal | -inf | -inf |
| inf | normal | normal |
| normal | inf | normal |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel_min(half* dst, half* x, half* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hmin(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_min(__gm__ half* dst, __gm__ half* x, __gm__ half* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hmin(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: nan, 0.5, 2.5, nan
y: -1.25, -0.25, 3.25, nan
result: -1.25, -0.25, 2.5, nan
```
