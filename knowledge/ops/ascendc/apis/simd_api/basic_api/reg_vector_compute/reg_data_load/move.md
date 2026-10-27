---
type: API
title: Move
description: 将srcReg中的有效元素逐个复制写入dstReg对应位置，无效位置保留dstReg原值，支持传入mask参数控制有效元素。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/Move.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h
---

# Move

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"`。

对srcReg中的有效元素逐个复制写入dstReg中对应位置处，无效位置保留dstReg原值。

如需将RegTensor复制到MaskReg，请参考接口[MaskGenWithRegTensor](mask_gen_with_reg_tensor_mask_reg.md)。

## 函数原型<a name="section620mcpsimp"></a>

- 支持传入mask参数：

    ```cpp
    template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::MERGING, typename U>
    __simd_callee__ inline void Move(U& dstReg, U& srcReg, MaskReg mask)
    ```

- 不支持传入mask参数：

    ```cpp
    template <typename T = DefaultType, typename U>
    __simd_callee__ inline void Move(U& dstReg, U& srcReg)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| mode | [MaskMergeMode](../aux_data_types/mask_merge_mode.md)枚举类型，选择MERGING模式或ZEROING模式。<br>&bull; ZEROING模式当前不支持。<br>&bull; MERGING模式下，mask未筛选的元素在dstReg中保留dstReg原值。 |
| U | 目的操作数的RegTensor类型，例如RegTensor\<half>，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcReg | 输入 | 源操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| mask | 输入 | 源操作数元素操作的有效指示，详细说明请参考[MaskReg](../register_data_types/mask_reg.md)。 |

## 数据类型


Ascend 950PR&950DT系列产品，目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。



## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

该接口的mode参数不支持ZEROING模式。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void MoveVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::Move(dstReg, srcReg, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```
