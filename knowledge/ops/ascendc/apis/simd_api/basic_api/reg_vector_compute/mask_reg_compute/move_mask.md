---
type: API
title: MoveMask
description: 从SetVectorMask设置的掩码寄存器中读取Mask值，按模板参数T对应格式转换后写入MaskReg。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/MaskReg_compute/MoveMask.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h
---

# MoveMask<a name="ZH-CN_TOPIC_0000001985457933"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h"`。

从[SetVectorMask](../../memory_vector_compute/mask_operations/set_vector_mask.md)设置的掩码寄存器{MASK1, MASK0}中读取Mask值，并按模板参数T对应的数据格式转换后写入返回值MaskReg。具体实现方式如下：

b16类型读取完整128bit的{MASK1, MASK0}，将每个bit复制为2bit；b32类型读取64bit的MASK0，并将每个bit复制为4bit。

![A2/A3与950 Mask机制对比](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_move_mask.png)

Mask与MaskReg的机制对比，具体请参考[A2/A3与950 Mask机制对比](../register_data_types/mask_reg.md)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T> 
__simd_callee__ inline MaskReg MoveMask()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 数据类型，决定mask值写入格式：<br>&bull; 对于b16类型，会读取完整的128bit {MASK1, MASK0}数据，并将每bit复制为2bit，写入函数返回值MaskReg。<br>&bull; 对于b32类型，会读取64bit MASK0数据，并将每bit复制为4bit，写入函数返回值MaskReg。 |

## 数据类型

支持的数据类型：b16、b32。

## 返回值说明<a name="section640mcpsimp"></a>

特殊寄存器中读取的MaskReg。

## 约束说明<a name="section633mcpsimp"></a>

- 本接口为兼容性接口，建议优先采用[CreateMask接口](../register_data_types/mask_reg.md)和[UpdateMask接口](../register_data_types/mask_reg.md)进行MaskReg计算。
- 使用前需要先调用[SetMaskCount](../../memory_vector_compute/mask_operations/set_mask_count.md)/[SetMaskNorm](../../memory_vector_compute/mask_operations/set_mask_norm.md)手动设置mask模式，并调用[SetVectorMask](../../memory_vector_compute/mask_operations/set_vector_mask.md)设置掩码寄存器SPR{MASK1, MASK0}。

## 调用示例<a name="section932512912207"></a>

```cpp
template <typename T>
__simd_vf__ inline void MoveMaskVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::MaskReg addMask = AscendC::Reg::MoveMask<T>();
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::Adds(srcReg, srcReg, 0, addMask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
    }
}
```
