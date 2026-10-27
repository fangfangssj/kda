---
type: API
title: Arange
description: 以传入标量值为起始或终止值，生成递增或递减的索引序列并保存在目的寄存器中。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/reg_vector_compute/index_operations/Arange.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/reg_compute/kernel_reg_compute_vec_arange_intf.h
---

# Arange<a name="ZH-CN_TOPIC_0000001960926493"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/reg_compute/kernel_reg_compute_vec_arange_intf.h"`。

递增模式时，该函数以传入的scalarValue的值为起始值，生成递增的索引；递减模式时，该函数以传入的scalarValue的值为终止值，生成递减的索引，并将索引保存在dstReg中。

**图 1**  Arange递增示意图

![Arange示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_arange_in.png)

**图 2**  Arange递减示意图

![Arange示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/reg_arange_de.png)

## 函数原型<a name="section520771712327"></a>

```cpp
template <typename T = DefaultType, IndexOrder order = IndexOrder::INCREASE_ORDER, typename U, typename S>
__simd_callee__ inline void Arange(S& dstReg, U scalarValue);
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名称 | 描述 |
| ------ | ------ |
| T | 目的操作数的数据类型。 |
| IndexOrder | 递增/递减模式。支持如下取值：<br>• INCREASE_ORDER：递增模式<br>• DECREASE_ORDER：递减模式 |
| U | 起始索引值的数据类型。 |
| S | 目的操作数的RegTensor类型，例如`RegTensor<half>`，由编译器自动推导，无需用户显式指定。 |

**表2**  函数参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| dstReg | 输出 | 目的操作数。<br>类型为[RegTensor](../register_data_types/reg_tensor.md)。 |
| scalarValue | 输入 | 源操作数。<br>类型为标量。 |

## 数据类型

支持的数据类型为：int8_t/int16_t/int32_t/half/float/int64_t。

## 约束说明<a name="section177921451558"></a>

对于int8_t/int16_t/int32_t/int64_t，当索引超过该数据类型的边界值时会翻转为负数，如对于int8_t，127之后为-128、-127，以此类推。

## 调用示例<a name="section642mcpsimp"></a>

```cpp
template<typename T>
__simd_vf__ inline void ArangeVF(__ubuf__ T* dstAddr, T scalarValue, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    mask = AscendC::Reg::CreateMask<T>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::Reg::Arange(dstReg, scalarValue);
        AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

- [Cat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/cat_950.md)
- [SwiGluGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/swi_glu_grad_950.md)
- [MhcSinkhorn 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_sinkhorn_950.md)
- [ChunkCat 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/chunk_cat_950.md)
- [Embedding 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/embedding_950.md)
- [MhcSinkhornBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_sinkhorn_backward_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [GatherNd 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/gather_nd_950.md)
- [MhcSinkhorn 950 Direct Invoke 算子知识](../../../../../operators/direct/transformer/mhc/mhc_sinkhorn_950.md)
- [ReplicationPad2d 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/replication_pad_2d_950.md)
