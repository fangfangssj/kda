---
type: API
title: Move
description: 将src中的元素复制到dst中，有mask时仅复制有效元素，无效元素填0。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/MaskReg_compute/Move.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h
---

# Move<a name="ZH-CN_TOPIC_0000001985577785"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h"`。

将src中的元素复制到dst中的对应位置。如果有输入mask，则仅复制被mask选定的有效元素，无效元素填0。

如需将RegTensor复制到MaskReg，请参考接口[MaskGenWithRegTensor](../reg_data_load/mask_gen_with_reg_tensor_mask_reg.md)。

## 函数原型<a name="section620mcpsimp"></a>

- 有输入mask

    ```cpp
    __simd_callee__ inline void Move(MaskReg& dst, MaskReg& src, MaskReg& mask)
    ```

- 无输入mask

    ```cpp
    __simd_callee__ inline void Move(MaskReg& dst, MaskReg& src)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 参数说明

| 参数名 | 描述 |
| --- | --- |
| dst | 目的操作数，类型为[MaskReg](../register_data_types/mask_reg.md)。 |
| src | 源操作数，类型为[MaskReg](../register_data_types/mask_reg.md)。 |
| mask | 指示在复制过程中哪些bit有效，类型为[MaskReg](../register_data_types/mask_reg.md)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section932512912207"></a>

```cpp
template <typename T>
__simd_vf__ inline void MoveVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    AscendC::Reg::MaskReg src = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALLF>();
    AscendC::Reg::MaskReg dst;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
        AscendC::Reg::Move(dst, src, mask);
        AscendC::Reg::Adds(srcReg, srcReg, 0, dst);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
    }
}
```
