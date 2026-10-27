---
type: API
title: StoreUnAlignMaskReg
description: 将数据从MaskReg连续搬运至非对齐Unified Buffer，利用非对齐寄存器暂存跨边界数据。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_store/StoreUnAlign_MaskReg.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h
---

# MaskReg非对齐搬出（StoreUnAlign）<a name="ZH-CN_TOPIC_0000002554449396"></a>

## 功能说明

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"`。

StoreUnAlign能够实现数据从[MaskReg](../register_data_types/mask_reg.md)连续搬运至非对齐Unified Buffer（UB），利用非对齐寄存器UnalignRegForStore作为临时缓存区，暂存跨对齐边界的数据，从而实现高效的连续非对齐数据传输。

## 函数原型

```cpp
template <typename T>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, MaskReg& mask, UnalignRegForStore& ureg)

template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T*& dstAddr, UnalignRegForStore& ureg, int32_t postUpdateStride)
```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
|-----|-----|
| T | 目的操作数的数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| postMode | 用于控制是否使能UB地址会自动更新功能（post update），PostLiteral类型。具体参数说明请参考[PostLiteral](../aux_data_types/post_literal.md)。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|-----|-----|-----|
| dstAddr | 输入/输出 | 目的操作数，UB起始地址，不需要32字节对齐。b16数据类型时需要16字节对齐，b32数据类型时需要8字节对齐。 |
| mask | 输入 | 源操作数，类型为[MaskReg](../register_data_types/mask_reg.md)。 |
| ureg | 输入/输出 | 非对齐寄存器，UnalignRegForStore类型，用于存储非32B对齐的数据，寄存器大小为32字节。 |
| postUpdateStride | 输入 | 地址更新步长，uint32_t类型，单位：元素个数。根据postMode的取值有两种情况：<br>&bull; POST_MODE_NORMAL：不支持。<br>&bull; POST_MODE_UPDATE：实际搬运UB起始地址为dstAddr，搬运后执行地址更新 dstAddr += postUpdateStride。 |

## 数据类型

目的操作数支持的数据类型为：b16、b32。

## 返回值说明

无

## 约束说明

- 数据类型为b16时，位于UB的地址必须16字节对齐，搬运完成后UB地址按16B偏移量更新，不需要手动更新地址。
- 数据类型为b32时，位于UB的地址必须8字节对齐，搬运完成后UB地址按8B偏移量更新，不需要手动更新地址。

## 关键特性说明

**非对齐数据搬出原理**

由[MaskReg原理](../register_data_types/mask_reg.md)可知，当操作数类型为b16时，每一个element对应2bit的MaskReg，且仅2bit中的最低位是有效的；当操作数类型为b32时，每一个element对应4bit的MaskReg，且仅4bit中的最低位是有效的。

将源寄存器MaskReg（32B）搬出至非对齐UB：

- UB地址上数据类型为b16时，硬件指令从每2bit数据中提取最低有效位（LSB），将MaskReg中32B数据打包成16B，写入UB。搬运完成后UB地址按16B偏移量更新。
- UB地址上数据类型为b32时，硬件指令从每4bit数据中提取最低有效位（LSB），将MaskReg中32B数据打包成8B，写入UB。搬运完成后UB地址按8B偏移量更新。

## 调用示例

```cpp
template <typename T>
__simd_vf__ inline void StoreUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint16_t oneRepeatSize, uint16_t offset, uint16_t repeatTimes)
{
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore ureg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::Reg::LoadAlign(mask, srcAddr + i * oneRepeatSize);
        // MaskReg非对齐搬出会自动进行地址偏移
        // 数据类型为b16时，32字节打包成16字节，偏移为16字节。
        // 数据类型为b32时，32字节打包成8字节，偏移为8字节。
        AscendC::Reg::StoreUnAlign(dstAddr, mask, ureg);
    }
    AscendC::Reg::StoreUnAlignPost(dstAddr, ureg, 0);
}
```

