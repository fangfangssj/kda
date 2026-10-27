---
type: API
title: GetCtrlSpr
description: 读取CTRL寄存器特定比特位上的值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.system_variable
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/special_register_access/GetCtrlSpr_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# GetCtrlSpr(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```cpp
template <int8_t startBit, int8_t endBit>
__aicore__ static inline int64_t GetCtrlSpr()
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| startBit | 起始比特位索引。 |
| endBit | 终止比特位索引。 |

## 数据类型

本接口为寄存器访问接口，不涉及张量数据类型操作。返回值类型固定为int64_t。

## 返回值说明

CTRL寄存器对应比特位上的值。

## 约束说明

- startBit和endBit的取值范围为[0, 63]。
- startBit小于等于endBit。
- 各比特位的说明请参考[SetCtrlSpr(ISASI)参数说明](set_ctrl_spr_isasi.md#参数说明)。

## 调用示例

如下为读取CTRL[48]比特位数值的示例。

```cpp
int64_t valueCtrlAfterSet = AscendC::GetCtrlSpr<48, 48>();
```

更多示例请参考[CtrlSpr样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/09_utils/ctrl_spr)。

- [GeluQuant 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/gelu_quant_950.md)
- [MlaProlog A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/mla_prolog.md)
- [SwigluGroupQuant 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/swiglu_group_quant_950.md)
- [MlaProlog 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/mla_prolog_950.md)
- [InplaceQuantScatter 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_quant_scatter_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [AddRmsNorm 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/add_rms_norm_950.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
