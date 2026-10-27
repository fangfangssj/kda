---
type: API
title: Hlog10
description: 计算bfloat16_t输入数据以10为底的对数。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_math_functions/hlog10.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# hlog10

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取以10为底，输入数据的对数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002576421049.png)

## 函数原型

```cpp
inline bfloat16_t hlog10(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

以10为底的x的对数。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| inf | inf | ASCRT_MAX_NORMAL_BF16（饱和到bfloat16_t最大有限值） |
| -inf, nan | nan | 0（nan被饱和为0） |
| 0 | -inf | -ASCRT_MAX_NORMAL_BF16 |
| x小于0 | nan | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelLog10(bfloat16_t* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hlog10(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelLog10(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hlog10(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0, 0.25, 4.0
result: -0.300781, 0.0, 0.300781, 1.0, 2.0, 3.0, -0.601563, 0.601563
```
