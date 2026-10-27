---
type: API
title: __hmin_nan
description: 获取两个输入数据中的最小值。任一输入为nan时，结果为nan。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_compare_functions/__hmin_nan.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hmin\_nan

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

获取两个输入数据中的最小值。任一输入为nan时，结果为nan。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002574049837.png)

## 函数原型

```cpp
bfloat16_t __hmin_nan(const bfloat16_t x, const bfloat16_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据中的最小值。本接口不受全局饱和模式影响，特殊值如下：

-   x为-0，y为+0时，返回x。
-   x为+0，y为-0时，返回y。
-   x，y任意一个为nan时，返回值为nan。
-   x，y任意一个为-inf时，返回值为-inf。
-   x为inf时，返回值为y。
-   y为inf时，返回值为x。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelHmin_nan(bfloat16_t* dst, bfloat16_t* x, bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hmin_nan(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHmin_nan(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hmin_nan(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.0, 0.5, 2.0, 2.0, nan, 1.0, inf, -inf
y: 0.5, 0.5, 2.0, 3.0, 1.0, nan, inf, 0.0
result: -1.0, 0.5, 2.0, 2.0, nan, nan, inf, -inf
```
