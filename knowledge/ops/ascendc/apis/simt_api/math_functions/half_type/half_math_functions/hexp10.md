---
type: API
title: Hexp10
description: 计算half输入x的10的x次方。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_math_functions/hexp10.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# hexp10

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

指定输入x，获取10的x次方。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002533480319.png)

## 函数原型

```cpp
inline half hexp10(half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

10的x次方。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| ±0 | 1 | 1 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | 0 | 0 |
| nan | nan | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void KernelExp10(half* dst, half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hexp10(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelExp10(__gm__ half* dst, __gm__ half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hexp10(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
result: 0.0562439, 3.16211, 316.25, 0.00017786
```
