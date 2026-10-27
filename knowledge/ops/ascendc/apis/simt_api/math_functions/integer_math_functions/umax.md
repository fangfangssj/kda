---
type: API
title: umax（整数数学函数）
description: 获取两个输入数据中的最大值。 !
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/umax.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/math_functions.h
---

# umax

## 功能说明

头文件路径为：`"simt_api/math_functions.h"`。

获取两个输入数据中的最大值。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002521590968.png)

## 函数原型

```cpp
inline unsigned int umax(const unsigned int x, const unsigned int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

两个输入数据中的最大值。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __global__ __launch_bounds__(1024) void KernelUmax(unsigned int* dst, unsigned int* x, unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = umax(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/math_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelUmax(__gm__ unsigned int* dst, __gm__ unsigned int* x, __gm__ unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = umax(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0, 12, 23, 70
y: 5, 0, 15, 7
result: 5, 12, 23, 70
```
