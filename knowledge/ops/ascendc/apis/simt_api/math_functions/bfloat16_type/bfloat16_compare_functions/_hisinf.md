---
type: API
title: __hisinf
description: 判断bfloat16类型浮点数是否为无穷，输入为inf或-inf时返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_compare_functions/__hisinf.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hisinf

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

判断浮点数是否为无穷。

## 函数原型

```cpp
inline bool __hisinf(bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

接口功能不受全局饱和模式影响。返回值为：

-   false：输入不是无穷。
-   true：输入为inf、-inf。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelIsInf(bool* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hisinf(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelIsInf(__gm__ bool* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hisinf(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, -0.5, -0.0, 0.0, inf, -inf, nan, 1.5
result: false, false, false, false, true, true, false, false
```
