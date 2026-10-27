---
type: API
title: Load
description: 从UB非32字节对齐的源地址搬运数据至RegTensor，搬运量为VL（256B），内部封装了LoadUnAlignPre和LoadUnAlign接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_load/Load.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h
---

# Load<a name="ZH-CN_TOPIC_0000002581346834"></a>

## 功能说明<a name="section12921612161519"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"`。

Reg矢量计算数据搬运接口，支持从Unified Buffer（UB）非32字节对齐的源地址srcAddr搬运至RegTensor。当RegTensor模板参数regTrait为RegTraitNumOne时，搬运量为VL（256B）；当regTrait为RegTraitNumTwo时，搬运量为2*VL（512B）。连续搬运时，用户需手动更新srcAddr地址。

该接口封装了[LoadUnAlignPre和LoadUnAlign](load_un_align_continuous.md)。

## 函数原型<a name="section429122319217"></a>

```cpp
template <typename T = DefaultType, typename U>
__simd_callee__ inline void Load(U& dstReg, __ubuf__ T* srcAddr)
```

## 参数说明<a name="section99612571362"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 目的操作数的RegTensor类型，例如RegTensor\<half>，由编译器自动推导，用户不需要手动填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstReg | 输出 | 目的操作数，类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| srcAddr | 输入 | 源操作数，UB起始地址，不需要32字节对齐。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。

- 当RegTensor模板参数regTrait为RegTraitNumOne时，支持的数据类型为：b8、b16、b32、b64。
- 当RegTensor模板参数regTrait为RegTraitNumTwo时，支持的数据类型为：complex32、b64。

## 返回值说明

无

## 约束说明

- 当RegTensor模板参数regTrait为RegTraitNumTwo时，srcAddr起始位置需要保证至少有2*VL（512B）可读数据。
- 接口内部定义了一个[UnalignRegForLoad](../register_data_types/unalign_reg_for_load_unalign_reg_for_store.md)，该寄存器数量上限为4，超出后编译器将报错。

## 调用示例

```cpp
template<typename T>
__simd_vf__ inline void LoadStoreVF(
    __ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t count, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::Load(srcReg, srcAddr + i * count);
        AscendC::Reg::Store(dstAddr + i * count, srcReg);
    }
}
```

- [MhcPreBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_backward_950.md)
