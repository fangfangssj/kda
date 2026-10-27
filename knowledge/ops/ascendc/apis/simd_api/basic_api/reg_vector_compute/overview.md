---
type: API
title: Reg矢量计算概述
description: Reg矢量计算API概述，直接操作Vector计算寄存器实现数据搬运与计算，较Memory矢量计算API提供更大开发自由度。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 概述

Reg矢量计算API是面向Reg矢量计算架构开发的API，用户可以通过该API直接对芯片中涉及Vector计算的寄存器进行操作，实现更大的灵活性和更好的性能。Reg矢量计算API与Memory矢量计算API功能相似，但与Memory矢量计算API输入和输出数据必须为LocalTensor不同，Reg矢量计算API的输入或输出数据均为Reg矢量计算寄存器。对于计算类API，其功能是从给定的寄存器获取数据，进行计算，并将结果保存在给定的寄存器。对于搬运类API，其功能是实现Unified Buffer（UB）和寄存器的数据搬运。由此可见，Reg矢量计算API相较于Memory矢量计算API，将数据搬运和Reg计算过程交给用户自主控制，从而实现更大的开发自由度。

## 如何使用Reg矢量计算API

基于寄存器的编程模型是指将数据从LocalTensor通过Reg搬运指令（Reg数据搬入）加载到寄存器中，进行复杂的数学计算Compute后通过Reg搬运指令（Reg数据搬出）搬出到LocalTensor中，所有的计算逻辑均在寄存器中完成，从而减少LocalTensor间的数据搬运，大大提升了整体性能，具体流程如下所示：

**图1**  Reg矢量计算  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_vector_overview.png "Reg矢量计算")

以[Reg矢量计算编程示例](#reg矢量计算编程示例)为例，完整的VectorFunction计算过程由以下几部分组成：
- 编写和调用VectorFunction（向量函数）。这些函数使用__simd_vf__标记，通过[asc\_vf\_call](vf_call/asc_vf_call.md)被调用；
- 定义矢量数据寄存器[RegTensor](register_data_types/reg_tensor.md)、掩码寄存器[MaskReg](register_data_types/mask_reg.md)；
- 编写for循环映射到硬件循环（[HardwareLoop](key_features/key_features.md#硬件循环hardwareloop)）；[UpdateMask](register_data_types/mask_reg.md#mask设置方式)函数用于更新参与计算的mask，每次循环都会消耗一个VL长度的元素。
- 循环内调用Reg数据搬入接口连续对齐搬入（[LoadAlign](reg_data_load/load_align_continuous.md)）从UB中搬入单个VL长度数据；
- 循环内调用Reg计算接口Add完成单次Repeat计算；
- 循环内调用Reg数据搬出接口连续对齐搬出（[StoreAlign](reg_data_store/store_align_continuous.md)）往UB中搬出计算后的数据；

## Reg矢量计算编程示例

```cpp
template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::Reg::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void Compute()
{
    // GM数据搬运至UB地址src0Addr及src1Addr
    ...
    // 调用vf函数
    asc_vf_call<AddVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize, repeatTimes);
    // UB地址dstAddr数据搬运至GM
    ...
}
```

>[!NOTE]说明 
>- GM与UB间的数据搬运需通过[GM与UB数据搬运](../memory_vector_compute/data_move/index.md)完成；
>- Vector Function的流水类型为PIPE_V，Vector Function内部的同步依赖通过[LocalMemBar](sync_control/local_mem_bar.md)接口插入。
