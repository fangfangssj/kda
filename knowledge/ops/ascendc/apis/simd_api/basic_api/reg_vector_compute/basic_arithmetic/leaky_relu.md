---
type: API
title: LeakyRelu
description: 按元素执行Leaky ReLU操作，大于0的元素直接写入目的操作数，否则乘以标量scalarValue后写入。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/basic_arithmetic/LeakyRelu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h
---

# LeakyRelu<a name="ZH-CN_TOPIC_0000002043279360"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径：`"basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h"`。

该接口用于按元素执行Leaky ReLU（Leaky Rectified Linear Unit）操作：当源操作数中某元素大于0时，直接将该元素写入目的操作数；否则将该元素乘以标量值scalarValue后写入目的操作数。计算公式如下：

$$
dstReg_i = \begin{cases} srcReg_i & srcReg_i > 0 \\ srcReg_i \times scalarValue & srcReg_i \leq 0 \end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void LeakyRelu(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask)
```

## 参数说明<a name="section389364115374"></a>

**表 1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 标量源操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式下，mask未筛选的元素在dstReg中置零。<br>&bull; MERGING模式当前不支持。 |
| S | 源操作数和目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。<br>源操作数的数据类型需要与目的操作数保持一致。 |
| scalarValue | 输入 | 源操作数。类型为标量，数据类型需与目的操作数保持一致。 |
| mask | 输入 | 源操作数中元素有效性的指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：half、float。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section232816306478"></a>

```cpp
template<typename T>
__simd_vf__ inline void LeakyReluVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, T scalarValue, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::LeakyRelu(dstReg, srcReg, scalarValue, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```
