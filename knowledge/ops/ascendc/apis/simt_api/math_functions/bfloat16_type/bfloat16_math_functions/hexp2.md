---
type: API
title: Hexp2
description: 计算bfloat16_t输入x的2的x次方。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_math_functions/hexp2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# hexp2

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

指定输入x，获取2的x次方。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002576421045.png)

## 函数原型

```cpp
inline bfloat16_t hexp2(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

2的x次方。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| inf | inf | ASCRT_MAX_NORMAL_BF16（饱和到bfloat16_t最大有限值） |
| -inf | 0 | 0 |
| nan | nan | 0（nan被饱和为0） |
| 其他 | 2的x次方，当结果超出bfloat16_t最大有限值时，结果为inf | 2的x次方，当结果超出bfloat16_t最大有限值时，结果为ASCRT_MAX_NORMAL_BF16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelExp2(bfloat16_t* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hexp2(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelExp2(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = hexp2(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0
result: 0.125, 0.25, 0.5, 1.0, 1.41406, 2.0, 4.0, 8.0
```
