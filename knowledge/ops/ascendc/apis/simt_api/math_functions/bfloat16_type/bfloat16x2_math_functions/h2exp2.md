---
type: API
title: H2Exp2
description: 指定输入x，对x的各元素，获取2的该元素次方。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# h2exp2

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

指定输入x，对x的各元素，获取2的该元素次方。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002576421061.png)

## 函数原型

```cpp
inline bfloat16x2_t h2exp2(bfloat16x2_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入的各元素，2的该元素次方。本接口受全局饱和模式影响，特殊值如下：

| 输入 | 非饱和模式返回值 | 饱和模式返回值 |
|---|---|---|
| 0 | 1.0 | 1.0 |
| -0 | 1.0 | 1.0 |
| nan | nan | 0 |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | 0 | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelExp2(bfloat16x2_t* dst, bfloat16x2_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2exp2(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelExp2(__gm__ bfloat16x2_t* dst, __gm__ bfloat16x2_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = h2exp2(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (-2, 1.25), (0.75, -0.25)
result: (0.419922, 1.41406), (5.65625, 0.0742188), (0.25, 2.375), (1.67969, 0.839844)
```
