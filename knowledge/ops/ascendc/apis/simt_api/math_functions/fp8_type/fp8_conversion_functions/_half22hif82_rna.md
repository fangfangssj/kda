---
type: API
title: __half22hif82_rna
description: 将half2类型数据的两个分量遵循CAST_ROUND模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/fp8_type/fp8_conversion_functions/__half22hif82_rna.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp8.h
---

# \_\_half22hif82\_rna

## 功能说明

头文件路径为：`"simt_api/asc_fp8.h"`。

将half2类型数据的两个分量遵循CAST\_ROUND模式转换为hifloat8\_t精度，返回转换后的hifloat8x2\_t类型数据。

## 函数原型

```cpp
inline hifloat8x2_t __half22hif82_rna(const half2 x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

将输入的两个分量遵循CAST\_ROUND模式转换成的hifloat8x2\_t类型数据。本接口受全局饱和模式影响，特殊值如下：

| x分量值 | 非饱和模式返回值 | 饱和模式返回值 |
|---|---|---|
| ±0 | 0 | 0 |
| nan | nan | +0 |
| inf | inf | 32768 |
| -inf | -inf | -32768 |
| ASCRT_MAX_NORMAL_FP16 | inf | 32768 |
| -ASCRT_MAX_NORMAL_FP16 | -inf | -32768 |
| ASCRT_MIN_DENORM_FP16 | 0 | 0 |

## 约束说明

SIMT编程场景当前不支持使用该接口。

## 调用示例

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp8.h"

    // 使用短向量可提升数据搬运效率
    __simt_vf__ __launch_bounds__(1024) inline void simt_half22hif82_rna(__gm__ half2* input, __gm__ hifloat8x2_t* output, uint32_t input_total_length)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        // 每个线程处理1个half2类型的数据，即2个half类型的数据，因此idx >= input_total_length / 2的线程不处理数据
        if (idx >= input_total_length / 2) {
            return;
        }
        output[idx] = __half22hif82_rna(input[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ half* input,  __gm__ uint8_t* output, uint32_t input_total_length)
    {
        asc_vf_call<simt_half22hif82_rna>(dim3(1024), (__gm__ half2*)input, (__gm__ hifloat8x2_t*)output, input_total_length);
    }
    ```

输入输出示例如下：

```text
x(half2)：(0.0, 17.0), (41.0, 9.0), (-41.0, 8.0), (nan, inf)
result(hifloat8x2_t)：(0, 16), (40, 9), (-40, 8), (nan, inf)
```
