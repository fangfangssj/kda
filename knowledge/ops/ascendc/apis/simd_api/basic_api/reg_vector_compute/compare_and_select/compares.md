---
type: API
title: Compares
description: 逐元素比较RegTensor中每个元素与标量的大小关系，结果为真则MaskReg对应比特位置1。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/compare_and_select/Compares.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_cmpsel_intf.h
---

# Compares<a name="ZH-CN_TOPIC_0000001956986933"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_vec_cmpsel_intf.h"`。

该接口用于逐元素比较源操作数中每个元素与一个标量的大小关系，根据模板参数指定的比较模式（LT/GT/GE/EQ/NE/LE）判断结果是否为真，若为真则输出`MaskReg`对应比特位置1，否则置0。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename U, typename S>
__simd_callee__ inline void Compares(MaskReg& dst, U& srcReg, S scalarValue, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :----- | :--- |
| T | 矢量源操作数数据类型。 |
| mode | 比较模式。支持如下取值：<br>• LT：小于<br>• GT：大于<br>• GE：大于或等于<br>• EQ：等于<br>• NE：不等于<br>• LE：小于或等于 |
| U | 矢量源操作数的`RegTensor`类型，由编译器自动推导，用户不需要填写。 |
| S | 标量操作数数据类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------- | :--- |
| dst | 输出 | [MaskReg](../register_data_types/mask_reg.md)类型，目的操作数。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| scalarValue | 输入 | 源操作数。<br>类型为标量。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 数据类型

**表3**  数据类型组合

| srcReg和scalarValue数据类型 | dst数据类型 |
| :-------------------------- | :---------- |
| int8_t | MaskReg |
| uint8_t | MaskReg |
| int16_t | MaskReg |
| uint16_t | MaskReg |
| half | MaskReg |
| bfloat16_t | MaskReg |
| int32_t | MaskReg |
| uint32_t | MaskReg |
| float | MaskReg |
| int64_t | MaskReg |
| uint64_t | MaskReg |

## 约束说明<a name="section633mcpsimp"></a>

通过mask参数控制的未选中元素在目的操作数中被置零。

## 调用示例<a name="section12223121385116"></a>

```cpp
template<typename T>
__simd_vf__ inline void ComparesVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, T scalarValue, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg cmpMaskReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);       
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(cmpMaskReg, srcReg0, scalarValue, mask);
        AscendC::Reg::Select(dstReg, srcReg0, srcReg1, cmpMaskReg);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [All 950 ACLNN 算子知识](../../../../../operators/aclnn/math/math/all_950.md)
