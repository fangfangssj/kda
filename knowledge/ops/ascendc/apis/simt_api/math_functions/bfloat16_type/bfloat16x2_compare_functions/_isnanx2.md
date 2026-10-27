---
type: API
title: __isnanx2
description: 判断bfloat16x2_t类型数据的两个分量是否为nan。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16x2_compare_functions/__isnanx2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_isnanx2

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

判断bfloat16x2_t类型数据的两个分量是否为nan。

## 函数原型

```cpp
bfloat16x2_t __isnanx2(bfloat16x2_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

判断输入数据各分量是否为nan，为nan时对应分量结果为1.0，否则为0.0。特殊值如下：

| x分量值 | 返回值 |
|---|---|
| nan | 1.0 |
| ±0 | 0 |
| ±inf | 0 |
| ±ASCRT_MAX_NORMAL_BF16 | 0 |
| ASCRT_MIN_DENORM_BF16 | 0 |
| 正常值 | 0 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __global__ __launch_bounds__(1024) void simt_isnanx2(bfloat16_t* x, bfloat16_t* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        bfloat16x2_t* input = (bfloat16x2_t*)x;
        bfloat16x2_t* out = (bfloat16x2_t*)dst;
        out[idx] = __isnanx2(input[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_isnanx2(__gm__ bfloat16x2_t* x, __gm__ bfloat16x2_t* dst, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        dst[idx] = __isnanx2(x[idx]);
    }

    __global__ __vector__ void compute_kernel(__gm__ bfloat16_t* x, __gm__ bfloat16_t* dst, uint32_t input_total_length)
    {
        asc_vf_call<simt_isnanx2>(dim3(1024), (__gm__ bfloat16x2_t*)x, (__gm__ bfloat16x2_t*)dst, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (-1.25, 0.5), (2.5, -3.75), (nan, 1.25), (0.75, nan)
result: (0, 0), (0, 0), (1, 0), (0, 1)
```
