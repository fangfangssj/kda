---
type: API
title: __hmax_nan
description: 获取两个输入数据中的最大值。任一输入为nan时，结果为nan。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_compare_functions/__hmax_nan.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hmax\_nan

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取两个输入数据中的最大值。任一输入为nan时，结果为nan。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002573929855.png)

## 函数原型

```cpp
bfloat16_t __hmax_nan(const bfloat16_t x, const bfloat16_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据中的最大值。本接口不受全局饱和模式影响，特殊值如下：

-   `x`为+0，`y`为-0时，返回`x`。
-   `x`为-0，`y`为+0时，返回`y`。
-   `x`，`y`任意一个为nan时，返回值为nan。
-   `x`，`y`任意一个为inf时，返回值为inf。
-   `x`为-inf时，返回值为`y`。
-   `y`为-inf时，返回值为`x`。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelHmax_nan(bfloat16_t* dst, bfloat16_t* x, bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hmax_nan(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHmax_nan(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hmax_nan(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.0, 0.5, 2.0, 2.0, nan, 1.0, inf, -inf
y: 0.5, 0.5, 2.0, 3.0, 1.0, nan, inf, 0.0
result: 0.5, 0.5, 2.0, 3.0, nan, nan, inf, 0.0
```
