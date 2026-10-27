---
type: API
title: __float2bfloat16
description: 遵循CAST_RINT模式，将浮点数转换为bfloat16精度，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__float2bfloat16.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_float2bfloat16

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

遵循CAST\_RINT模式，将浮点数转换为bfloat16精度，返回转换后的值。

## 函数原型

```cpp
inline bfloat16_t __float2bfloat16(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入遵循CAST\_RINT模式转换成的bfloat16类型数据。本接口受全局饱和模式影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
|---|---|---|
| 0 | 0 | 0 |
| -0 | -0 | -0 |
| nan | nan | 0 |
| inf | inf | ASCRT_MAX_NORMAL_BF16 |
| -inf | -inf | -ASCRT_MAX_NORMAL_BF16 |
| ＞ASCRT_MAX_NORMAL_BF16 | inf | ASCRT_MAX_NORMAL_BF16 |
| ＜-ASCRT_MAX_NORMAL_BF16 | -inf | -ASCRT_MAX_NORMAL_BF16 |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__float2bfloat16(bfloat16_t* dst, float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2bfloat16(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2bfloat16(__gm__ bfloat16_t* dst, __gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __float2bfloat16(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 128.25, 128.5, 128.75, -128.5, 0.0, -0.0, inf, nan
result: 128.0, 128.0, 129.0, -128.0, 0.0, -0.0, inf, nan
```

- [ForeachTan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tan_950.md)
- [ForeachAcos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_acos_950.md)
- [ForeachAddcdivScalar 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_addcdiv_scalar_950.md)
- [ForeachAsin 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_asin_950.md)
- [ForeachAtan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_atan_950.md)
- [ForeachErfc 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_erfc_950.md)
- [ForeachMulList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_mul_list_950.md)
- [ForeachTanh 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tanh_950.md)
- [ForeachAddcdivScalar 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/foreach/foreach_addcdiv_scalar_950.md)

## 配套 API 与常见组合

- [__bfloat162float](_bfloat162float.md)——AcosCompute<bfloat16_t> 同一特化内 bf16→fp32 计算→fp32→bf16 的输入/输出转换环，P(a|b)=1.00 佐证凡上变换必配下变换
