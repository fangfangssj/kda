---
type: API
title: BFloat16数学函数hexp
description: 指定输入x，获取e的x次方。 !
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_math_functions/hexp.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# hexp

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

指定输入x，获取e的x次方。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002545900860.png)

## 函数原型

```cpp
inline bfloat16_t hexp(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

e的x次方。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | 0 | 0 |
| nan | nan | 0 |
| 其他 | e的x次方，当结果超出bfloat16_t最大有限值时，结果为inf | e的x次方，当结果超出bfloat16_t最大有限值时，结果为ASCRT_MAX_NORMAL_BF16 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel_exp(bfloat16_t* dst, bfloat16_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hexp(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_exp(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hexp(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0
result: 0.0498047, 0.135742, 0.367188, 1.0, 1.64844, 2.71875, 7.375, 20.125
```
