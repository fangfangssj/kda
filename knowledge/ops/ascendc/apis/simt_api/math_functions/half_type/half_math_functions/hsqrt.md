---
type: API
title: Hsqrt
description: 计算half输入数据x的平方根。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_math_functions/hsqrt.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# hsqrt

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

获取输入数据x的平方根。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002533480327.png)

## 函数原型

```cpp
inline half hsqrt(half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，输入数据。 |

## 返回值说明

输入数据的平方根。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| ±0 | 0 | 0 |
| nan | nan | 0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | nan | 0 |
| x<0 | nan | 0 |

## 约束说明

本接口支持的输入数据范围为x大于等于0，否则非饱和模式下返回值为nan，饱和模式下返回值为0。

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel_sqrt(half* dst, half* x, uint32_t total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hsqrt(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_sqrt(__gm__ half* dst, __gm__ half* x, uint32_t total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hsqrt(x[idx]);
    }

    __global__ __vector__ void run_sqrt(__gm__ half* dst, __gm__ half* x, uint32_t total_length)
    {
        asc_vf_call<kernel_sqrt>(dim3(1024), dst, x, total_length);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
result: nan, 0.707031, 1.58105, nan
```
