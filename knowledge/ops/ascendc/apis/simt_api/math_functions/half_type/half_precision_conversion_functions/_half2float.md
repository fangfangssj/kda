---
type: API
title: __half2float
description: 将half类型数据转换为浮点数，返回转换后的值。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.type_conversion
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/half_type/half_precision_conversion_functions/__half2float.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/asc_fp16.h
---

# \_\_half2float

## 功能说明

头文件路径为：`"simt_api/asc_fp16.h"`。

将half类型数据转换为浮点数，返回转换后的值。

## 函数原型

```cpp
inline float __half2float(const half x)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数。 |

## 返回值说明

输入转换成的浮点数。特殊值如下：

| x值 | 返回值 |
| --- | --- |
| 0 | 0 |
| -0 | -0 |
| nan | nan |
| inf | inf |
| -inf | -inf |

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __global__ __launch_bounds__(1024) void kernel__half2float(float* dst, half* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __half2float(x[idx]);
    }
    ```

- SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/asc_fp16.h"

    __simt_vf__ __launch_bounds__(1024) inline void kernel__half2float(__gm__ float* dst, __gm__ half* x, uint32_t total_length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= total_length) {
            return;
        }
        dst[idx] = __half2float(x[idx]);
    }
    __global__ __vector__ void cast_kernel(__gm__ float* dst, __gm__ half* x, uint32_t total_length)
    {
        asc_vf_call<kernel__half2float>(dim3(1024), dst, x, total_length);
    }
    ```

输入输出示例如下：

```text
x: -2, -0.5, 0.5, 2, inf, -inf, nan
result: -2, -0.5, 0.5, 2, inf, -inf, nan
```

- [ForeachTan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_tan_950.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [ForeachAcos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_acos_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [ForeachAddList 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_add_list_950.md)
- [ForeachAddcdivScalar 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_addcdiv_scalar_950.md)
- [ForeachAsin 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_asin_950.md)
- [ForeachAtan 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_atan_950.md)
- [ForeachCos 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_cos_950.md)
- [ForeachCosh 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_cosh_950.md)
- [ForeachErfc 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/foreach/foreach_erfc_950.md)

## 配套 API 与常见组合

- [__float2half](_float2half.md)——foreach_acos 的 AcosCompute<half> 同一特化内 __half2float 下变换→acosf→__float2half 上变换回写，转换对生命周期配对，P(b|a)=1.00，缺一即类型断裂
