---
type: API
title: Mull
description: 根据mask对源操作数srcReg0、srcReg1按元素相乘，将乘法结果的低位部分写入dstReg0，溢出（高位）部分写入dstReg1。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/Mull.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h
---

# Mull<a name="ZH-CN_TOPIC_0000001929827664"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"`。

该接口根据mask对输入数据srcReg0、srcReg1按元素相乘操作，将乘法结果的低位部分写入dstReg0，溢出（高位）部分写入dstReg1。计算公式如下：

$$
dstReg0_i = (srcReg0_i \times srcReg1_i) mod2^{bit}
$$

$$
dstReg1_i = (srcReg0_i \times srcReg1_i) / 2^{bit}
$$

其中，bit表示操作数的位宽bit数。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_mull.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U>
__simd_callee__ inline void Mull(U& dstReg0, U& dstReg1, U& srcReg0, U& srcReg1, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg0 | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| dstReg1 | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg0 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg1 | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：int32_t、uint32_t。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void MullVF(__ubuf__ T* dst0Addr, __ubuf__ T* dst1Addr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg0;
    AscendC::Reg::RegTensor<T> dstReg1;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::Reg::Mull(dstReg0, dstReg1, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dst0Addr + i * oneRepeatSize, dstReg0, mask);
        AscendC::Reg::StoreAlign(dst1Addr + i * oneRepeatSize, dstReg1, mask);
    }
}
```

- [AvgPool2d 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/pooling/avg_pool2d_950.md)
