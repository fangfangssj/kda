---
type: API
title: Pack
description: 将源操作数元素的低位部分写入目的寄存器的低半部分或高半部分，实现数据位宽压缩。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/data_compression/Pack.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_pack_intf.h
---

# Pack<a name="ZH-CN_TOPIC_0000002010754854"></a>

## 功能说明<a name="section16390191715619"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_pack_intf.h"`。

将源操作数srcReg中的元素选取低8位（对于b16类型）、低16位（对于b32类型）、低32位（对于b64类型）写入dstReg的低半部分或高半部分。

**图 1**  Pack示意图

![Pack](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_pack_1.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename S, typename V>
__simd_callee__ inline void Pack(S& dstReg, V& srcReg)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名称 | 描述 |
| ------ | ------ |
| T | 目的操作数数据类型。<br> 源操作数和目的操作数的数据类型约束参考见[表 源操作数和目的操作数的数据类型对应表](#table3)。|
| U | 源操作数数据类型。 |
| part | 枚举类型，用于控制写入dstReg的低半部分还是高半部分。<br>• HighLowPart::LOWEST，低位模式，写入dstReg的低半部分。<br>• HighLowPart::HIGHEST，高位模式，写入dstReg的高半部分。<br> 注：RegTraitNumTwo只支持LOWEST模式。|
| S | 目的操作数RegTensor类型，由编译器自动推导，用户不需要填写。 |
| V | 源操作数RegTensor类型，由编译器自动推导，用户不需要填写。 |

**表2**  函数参数说明

| 参数名称 | 描述 |
| ------ | ------ |
| dstReg | 目的操作数。<br> 类型为[RegTensor](../register_data_types/reg_tensor.md)。|
| srcReg | 源操作数。<br> 类型为[RegTensor](../register_data_types/reg_tensor.md)。|

## 返回值说明<a name="section640mcpsimp"></a>

无

## 数据类型

**表 3**  源操作数和目的操作数的数据类型对应表<a id="table3"></a>

| T数据类型 | U数据类型 |
| ------ | ------ |
| uint8_t | int16_t |
| uint8_t | uint16_t |
| uint16_t | int32_t |
| uint16_t | uint32_t |
| uint32_t | int64_t |
| uint32_t | uint64_t |

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section932512912207"></a>

```cpp
template<typename T, typename U, int32_t mode = 0>
__simd_vf__ inline void PackVF(__ubuf__ T* dstAddr, __ubuf__ U* srcAddr, uint32_t oneDstRepSize, uint32_t oneSrcRepSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<U> srcReg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::LoadAlign(srcReg, srcAddr + i * oneSrcRepSize);
        if constexpr (mode == 0) {
            AscendC::Reg::Pack<T, U, AscendC::Reg::HighLowPart::LOWEST>(dstReg, srcReg);
        } else if constexpr (mode == 1) {
            AscendC::Reg::Pack<T, U, AscendC::Reg::HighLowPart::HIGHEST>(dstReg, srcReg);
        }
        AscendC::Reg::StoreAlign(dstAddr + i * oneDstRepSize, dstReg, mask);
    }
}
```
