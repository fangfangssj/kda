---
type: API
title: __hisnan
description: 判断浮点数是否为nan。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_compare_functions/__hisnan.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hisnan

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

判断浮点数是否为nan。

## 函数原型

```cpp
inline bool __hisnan(half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

返回是否为nan的结果。输入为非nan返回false，输入为nan返回true。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void KernelIsNan(bool* dst, half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hisnan(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelIsNan(__gm__ bool* dst, __gm__ half* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __hisnan(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: -1.25, nan, inf, -inf
result: false, true, false, false
```

- [ForeachMinimumList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_minimum_list_950.md)
