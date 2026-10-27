---
type: API
title: GetVecLen
description: 获取RegTensor位宽VL（Vector Length）的大小。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetVecLen.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetVecLen

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取[RegTensor](../../reg_vector_compute/register_data_types/reg_tensor.md)位宽VL（Vector Length）的大小。

## 函数原型

```cpp
__aicore__ inline constexpr uint32_t GetVecLen()
```

## 参数说明

无

## 返回值说明

Vector Length的大小，返回常量256，单位为Byte。

## 约束说明

无

## 调用示例

如下样例通过GetVecLen获取循环迭代次数：

```cpp
template <typename T>
__aicore__ inline void AddCustomImpl(__local_mem__ T *dst, __local_mem__ T *src0, __local_mem__ T *src1,
    uint32_t calCount)
{
    AscendC::Reg::RegTensor<T> reg0;
    AscendC::Reg::RegTensor<T> reg1;
    AscendC::Reg::RegTensor<T> reg2;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t repeatElm = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTime = AscendC::CeilDivision(calCount, repeatElm);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(calCount);
        AscendC::Reg::LoadAlign(reg0, src0 + i * repeatElm);
        AscendC::Reg::LoadAlign(reg1, src1 + i * repeatElm);
        AscendC::Reg::Add(reg2, reg0, reg1, mask);
        AscendC::Reg::StoreAlign(dst + i * repeatElm, reg2, mask);
    }
}
```

- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [RotateQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/rotate_quant_950.md)
- [BlockSparseAttention 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/block_sparse_attention_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [AddLayerNormQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_layer_norm_quant_950.md)
- [NormRopeConcat 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/norm_rope_concat_950.md)
- [Cumsum 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/cumsum_950.md)
- [GroupNormSwishGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/group_norm_swish_grad_950.md)
- [Xlog1py 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/xlog1py_950.md)
- [LayerNorm 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/layer_norm_950.md)
