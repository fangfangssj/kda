---
type: API
title: __highs2bfloat162
description: 分别提取两个bfloat162输入的高16位，并填充到bfloat162中，返回填充后的数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__highs2bfloat162.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_highs2bfloat162

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

分别提取两个bfloat162输入的高16位，并填充到bfloat162中，返回填充后的数据。

## 函数原型

```cpp
inline bfloat16x2_t __highs2bfloat162(const bfloat16x2_t x, const bfloat16x2_t y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |
| y | 输入 | 源操作数。 |

## 返回值说明

提取两个bfloat162输入的高16位，并填充到bfloat162中的结果。特殊值如下：

| 输入高16位值 | 返回值 |
|---|---|
| ±0 | ±0 |
| nan | nan |
| ±inf | ±inf |
| ASCRT_MAX_NORMAL_BF16 | ASCRT_MAX_NORMAL_BF16 |
| -ASCRT_MAX_NORMAL_BF16 | -ASCRT_MAX_NORMAL_BF16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __aicore__ void simt_highs2bfloat162(bfloat16x2_t* input1, bfloat16x2_t* input2, bfloat16x2_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx > input_total_length / 2) {
            return;
        }
        output[idx] = __highs2bfloat162(input1[idx], input2[idx]);
    }
    __global__ __launch_bounds__(1024) void cast_kernel(bfloat16_t* input1,  bfloat16_t* input2, bfloat16_t* output, uint32_t input_total_length)
    {
        simt_highs2bfloat162((bfloat16x2_t*)input1, (bfloat16x2_t*)input2, (bfloat16x2_t*)output, input_total_length);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_highs2bfloat162(__gm__ bfloat16x2_t* input1, __gm__ bfloat16x2_t* input2, __gm__ bfloat16x2_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个bfloat16x2_t类型的数据，即2个bfloat16_t类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx > input_total_length / 2) {
            return;
        }
        output[idx] = __highs2bfloat162(input1[idx], input2[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ bfloat16_t* input1,  __gm__ bfloat16_t* input2, __gm__ bfloat16_t* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_highs2bfloat162>(dim3(1024), (__gm__ bfloat16x2_t*)input1, (__gm__ bfloat16x2_t*)input2, (__gm__ bfloat16x2_t*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x: (0.5, -1.0), (1.0, 2.0), (3.0, 4.0), (0.0, 3.0), (nan, 1.0), (1.0, 2.0), (inf, -inf), (nan, nan)
y: (1.0, 0.5), (1.0, 2.0), (2.0, 3.0), (1.0, 2.0), (0.0, 1.0), (1.0, nan), (inf, 0.0), (nan, nan)
result: (-1.0, 0.5), (2.0, 2.0), (4.0, 3.0), (3.0, 2.0), (1.0, 1.0), (2.0, nan), (-inf, 0.0), (nan, nan)
```
