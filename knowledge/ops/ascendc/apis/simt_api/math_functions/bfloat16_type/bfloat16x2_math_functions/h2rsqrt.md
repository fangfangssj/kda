---
type: API
title: H2Rsqrt
description: 获取输入数据x各元素的平方根的倒数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16x2_math_functions/h2rsqrt.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# h2rsqrt

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取输入数据x各元素的平方根的倒数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002576421069.png)

## 函数原型

```cpp
inline bfloat16x2_t h2rsqrt(bfloat16x2_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，输入数据。 |

## 返回值说明

输入数据各元素的平方根的倒数。本接口受全局饱和模式影响，特殊值如下：

| 输入 | 非饱和模式返回值 | 饱和模式返回值 |
|---|---|---|
| 0 | inf | ASCRT_MAX_NORMAL_BF16 |
| -0 | inf | ASCRT_MAX_NORMAL_BF16 |
| nan | nan | 0 |
| inf | 0 | 0 |
| -inf | nan | 0 |
| x＜0 | nan | 0 |

## 约束说明

本接口支持的输入数据范围为x大于等于0，否则返回值为nan。

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelRsqrt(bfloat16x2_t* dst, bfloat16x2_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2rsqrt(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelRsqrt(__gm__ bfloat16x2_t* dst, __gm__ bfloat16x2_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2rsqrt(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (-2, 1.25), (0.75, -0.25)
result: (nan, 1.41406), (0.632813, nan), (nan, 0.894531), (1.15625, nan)
```
