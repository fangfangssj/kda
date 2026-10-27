---
type: API
title: Hceil
description: 获取大于或等于输入bfloat16_t数据的最小整数值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_math_functions/hceil.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# hceil

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取大于或等于输入数据的最小整数值。

## 函数原型

```cpp
inline bfloat16_t hceil(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

大于或等于输入数据的最小整数值。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | -inf | -ASCRT_MAX_NORMAL_BF16 |
| nan | nan | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelCeil(bfloat16_t* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hceil(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelCeil(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hceil(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0
result: 0.0, 1.0, 0.0, 1.0, -1.0, 2.0, -1.0, 2.0
```
