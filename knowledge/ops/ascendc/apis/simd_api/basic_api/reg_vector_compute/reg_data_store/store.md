---
type: API
title: Store
description: 从RegTensor搬出至非32字节对齐的UB地址，支持搬出VL或指定count个数据。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/reg_data_store/Store.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h
---

# Store

## 功能说明

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"`。

Reg矢量计算数据搬运接口，支持从RegTensor搬出至非32字节对齐的Unified Buffer（UB）地址dstAddr。连续搬运时，用户需手动更新dstAddr地址。

该接口封装了[StoreUnAlign和StoreUnAlignPost](store_un_align_continuous.md)。

## 函数原型

-   搬运srcReg中保存的全部数据

    ```cpp
    template <typename T = DefaultType, typename U>
    __simd_callee__ inline void Store(__ubuf__ T* dstAddr, U& srcReg)
    ```

-   搬运count个数据

    ```cpp
    template <typename T = DefaultType, typename U>
    __simd_callee__ inline void Store(__ubuf__ T* dstAddr, U& srcReg, uint32_t count)
    ```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。支持的数据类型请参考[数据类型](#数据类型)。 |
| U | 源操作数的RegTensor类型，例如RegTensor\<half>，由编译器自动推导，用户不需要填写。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstAddr | 输出 | 目的操作数，UB起始地址，不需要32字节对齐。 |
| srcReg | 输入 | 源操作数，类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| count | 输入 | 搬运数据量，仅带count的重载需要配置。连续搬运时需手动更新地址：`dstAddr = dstAddr + count`。 |

## 数据类型

目的操作数与源操作数的数据类型需要保持一致。

- 当RegTensor模板参数regTrait为RegTraitNumOne时支持的数据类型为：b8、b16、b32、b64。
- 当RegTensor模板参数regTrait为RegTraitNumTwo时支持的数据类型为：complex32、b64。

## 返回值说明

无

## 约束说明

-   不带count的重载搬运srcReg中保存的全部数据：RegTraitNumOne搬运256B，RegTraitNumTwo搬运512B。dstAddr起始位置需要保证至少有对应大小的可写空间。
-   对于带count的重载，count不能大于srcReg能存储的数据个数，且dstAddr起始位置需要保证至少有`count * sizeof(T)`的可写空间：
    - 当RegTensor模板参数regTrait为RegTraitNumOne时，`count <= 256B / sizeof(T)`。
    - 当RegTensor模板参数regTrait为RegTraitNumTwo时，`count <= 512B / sizeof(T)`。
-   接口内部定义了一个[UnalignRegForStore](../register_data_types/unalign_reg_for_load_unalign_reg_for_store.md)，该寄存器数量上限为4，超出后编译器将报错。

## 调用示例

- 搬出srcReg中保存的全部数据

    ```cpp
    template<typename T>
    __simd_vf__ inline void LoadStoreVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t count, uint16_t repeatTimes)
    {
        AscendC::Reg::RegTensor<T> srcReg;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::Load(srcReg, srcAddr + i * count);
            AscendC::Reg::Store(dstAddr + i * count, srcReg);
        }
    }
    ```

- 搬出count个数据

    ```cpp
    template<typename T>
    __simd_vf__ inline void LoadStoreVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t count, uint16_t repeatTimes)
    {
        AscendC::Reg::RegTensor<T> srcReg;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::Load(srcReg, srcAddr + i * count);
            AscendC::Reg::Store(dstAddr + i * count, srcReg, count);
        }
    }
    ```

- [MhcPreBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_backward_950.md)
