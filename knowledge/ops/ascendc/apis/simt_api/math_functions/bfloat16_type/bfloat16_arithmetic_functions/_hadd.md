---
type: API
title: __hadd
description: 计算两个bfloat16类型数据相加的结果，并遵循CAST_RINT模式对结果进行舍入。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_arithmetic_functions/__hadd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_hadd

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

计算两个bfloat16类型数据相加的结果，并遵循CAST\_RINT模式对结果进行舍入。

## 函数原型

```cpp
bfloat16_t __hadd(const bfloat16_t x, const bfloat16_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

输入数据相加的结果。本接口不受全局饱和模式影响，特殊值如下：

-   \_\_hadd\(x, y\)等价于\_\_hadd\(y, x\)。
-   x为有限值，y为±inf时，返回值为±inf。
-   x为±inf，y为±inf时，返回值为±inf。
-   x为±inf，y为∓inf时，返回值为nan。
-   x为±0，y为±0时，返回值为±0。
-   对有限值x（包括±0），x=-y时，返回值为+0。
-   x，y任意一个为nan时，返回值为nan。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void KernelHadd(bfloat16_t* dst, bfloat16_t* x, bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hadd(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelHadd(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* x, __gm__ bfloat16_t* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hadd(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 4.0
y: 2.0, 0.5, -1.0, -1.0, 0.5, 2.0, 4.0, 0.5
result: 2.5, 0.0, 0.0, -2.0, 2.5, 0.0, 4.0, 4.5
```
