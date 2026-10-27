---
type: API
title: __bfloat162float
description: 将bfloat16类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/bfloat16_type/bfloat16_precision_conversion_functions/__bfloat162float.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_bf16.h
---

# \_\_bfloat162float

## 功能说明

头文件路径为：`"simt_api/asc_bf16.h"`。

将bfloat16类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __bfloat162float(const bfloat16_t x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入转换为浮点数的结果。特别场景说明如下：

| x值 | 返回值 |
|---|---|
| 0 | 0 |
| -0 | -0 |
| nan | nan |
| inf | inf |
| -inf | -inf |

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __global__ __launch_bounds__(1024) void kernel__bfloat162float(float* dst, bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __bfloat162float(x[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_bf16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__bfloat162float(__gm__ float* dst, __gm__ bfloat16_t* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __bfloat162float(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 0.5, -0.5, -0.0, 0.0, inf, -inf, nan, 1.5
result: 0.5, -0.5, -0.0, 0.0, inf, -inf, nan, 1.5
```

- [ForeachTan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tan_950.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [ForeachAcos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_acos_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [ForeachAddList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_add_list_950.md)
- [ForeachAddcdivScalar 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_addcdiv_scalar_950.md)
- [ForeachAsin 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_asin_950.md)
- [ForeachAtan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_atan_950.md)
- [ForeachDivList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_div_list_950.md)
- [ForeachErfc 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_erfc_950.md)
- [ForeachMaximumList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_maximum_list_950.md)

## 配套 API 与常见组合

- [__float2bfloat16](_float2bfloat16.md)——AcosCompute<bfloat16_t> 同一特化内 bf16→fp32 计算→fp32→bf16 的输入/输出转换环，P(a|b)=1.00 佐证凡上变换必配下变换
