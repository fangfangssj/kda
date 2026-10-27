---
type: API
title: __float2half
description: 遵循CAST_RINT模式，将浮点数转换为半精度浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__float2half.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_float2half

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

遵循CAST\_RINT模式，将浮点数转换为半精度浮点数，返回转换后的值。

## 函数原型

```cpp
inline half __float2half(const float x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

将float类型数据按照CAST\_RINT模式转换为half类型数据。本接口受全局饱和寄存器的影响，特殊值如下：

| x值 | 非饱和模式返回值 | 饱和模式返回值 |
| --- | --- | --- |
| ±0 | ±0 | ±0 |
| nan | nan | ±0 |
| inf | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| -inf | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |
| x>ASCRT\_MAX\_NORMAL\_FP16 | inf | ASCRT\_MAX\_NORMAL\_FP16 |
| x<-ASCRT\_MAX\_NORMAL\_FP16 | -inf | -ASCRT\_MAX\_NORMAL\_FP16 |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__float2half(half* dst, float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2half(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__float2half(__gm__ half* dst, __gm__ float* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __float2half(x[idx]);
    }
    ```

输入输出示例如下：

```text
x: 128, 128.0625, 128.1875, -128.1875, inf, -inf, nan
result: 128, 128, 128.25, -128.25, inf, -inf, nan
```

- [ForeachTan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tan_950.md)
- [ForeachAcos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_acos_950.md)
- [ForeachAddcdivScalar 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_addcdiv_scalar_950.md)
- [ForeachAsin 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_asin_950.md)
- [ForeachAtan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_atan_950.md)
- [ForeachCos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_cos_950.md)
- [ForeachCosh 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_cosh_950.md)
- [ForeachErfc 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_erfc_950.md)
- [ForeachMulList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_mul_list_950.md)
- [ForeachTanh 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tanh_950.md)
- [ForeachAddcdivScalar 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/foreach/foreach_addcdiv_scalar_950.md)

## 配套 API 与常见组合

- [__half2float](_half2float.md)——foreach_acos 的 AcosCompute<half> 同一特化内 __half2float 下变换→acosf→__float2half 上变换回写，转换对生命周期配对，P(b|a)=1.00，缺一即类型断裂
