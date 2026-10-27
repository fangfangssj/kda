---
type: API
title: __hlt
description: 比较两个half类型数据，仅当第一个数小于第二个数时，返回true。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_compare_functions/__hlt.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_hlt

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

比较两个half类型数据，仅当第一个数小于第二个数时，返回true。

## 函数原型

```cpp
bool __hlt(half x, half y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

比较输入的第一个数是否小于第二个数的结果。任一输入为nan时返回false。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel_hlt(bool* dst, half* x, half* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hlt(x[idx], y[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_hlt(__gm__ bool* dst, __gm__ half* x, __gm__ half* y, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __hlt(x[idx], y[idx]);
    }

    __global__ __vector__ void run_kernel_hlt(__gm__ bool* dst, __gm__ half* x, __gm__ half* y, uint32_t total_length)
    {
        asc_vf_call<kernel_hlt>(dim3(1024), dst, x, y, total_length);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, nan
y: -0.5, 0.5, 1.25, nan
result: true, false, false, false
```
