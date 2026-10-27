---
type: API
title: H2Sin
description: 获取输入数据各元素的三角函数正弦值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half2_math_functions/h2sin.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# h2sin

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

获取输入数据各元素的三角函数正弦值。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002513174474.png)

## 函数原型

```cpp
inline half2 h2sin(half2 x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入数据各元素的三角函数正弦值。本接口受全局饱和寄存器影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| 0 | 0 | 0 |
| -0 | -0 | -0 |
| nan | nan | 0 |
| ±inf | nan | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void KernelSin(half2* dst, half2* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2sin(x[idx]); // 对src源地址的第idx个元素取三角函数正弦值
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelSin(__gm__ half2* dst, __gm__ half2* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2sin(x[idx]); // 对src源地址的第idx个元素取三角函数正弦值
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (-2, 1.25), (0.75, -0.25)
result: (-0.949219, 0.479492), (0.598633, 0.571777), (-0.90918, 0.949219), (0.681641, -0.247437)
```
