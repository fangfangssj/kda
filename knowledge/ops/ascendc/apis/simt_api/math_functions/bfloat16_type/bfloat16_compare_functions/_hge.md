---
type: API
title: __hge
description: 比较两个bfloat16类型数据，仅当第一个数据大于或等于第二个数据时，返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_compare_functions/__hge.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hge

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

比较两个bfloat16类型数据，仅当第一个数据大于或等于第二个数据时，返回true。

## 函数原型

```cpp
bool __hge(bfloat16_t x, bfloat16_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入的第一个数是否大于或等于第二个数的结果。

-   true：第一个数大于或等于第二个数。
-   false：第一个数小于第二个数，或者任一输入为nan。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelHge(bool* dst, bfloat16_t* x, bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hge(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHge(__gm__ bool* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hge(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.0, 0.5, 2.0, 2.0, nan, 1.0, inf, -inf
y: 0.5, 0.5, 2.0, 3.0, 1.0, nan, inf, 0.0
result: false, true, true, false, false, false, true, false
```
