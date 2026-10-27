---
type: API
title: Hfloor
description: 获取小于或等于输入half数据的最大整数值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_math_functions/hfloor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# hfloor

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

获取小于或等于输入数据的最大整数值。

## 函数原型

```cpp
inline half hfloor(half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

小于或等于输入数据的最大整数值。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| ±0 | 0 | 0 |
| nan | nan | 0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel_hfloor(half* dst, half* x, uint32_t total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hfloor(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel_hfloor(__gm__ half* dst, __gm__ half* x, uint32_t total_length)
    {
        uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = hfloor(x[idx]);
    }

    __global__ __vector__ void run_hfloor(__gm__ half* dst, __gm__ half* x, uint32_t total_length)
    {
        asc_vf_call<kernel_hfloor>(dim3(1024), dst, x, total_length);
    }
    ```

输入输出示例如下：

```text
x: -1.25, 0.5, 2.5, -3.75
result: -2, 0, 2, -4
```
