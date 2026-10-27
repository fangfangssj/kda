---
type: API
title: Mask接口汇总（附录）
description: 汇总 Atlas A2、A3 系列接口的 isSetMask 参数支持情况及 Mask 设置方式。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/Mask_interface_summary.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Mask接口汇总


以下内容针对如下型号生效：

- Atlas A3系列产品


- Atlas A2系列产品


本节汇总了各接口中模板参数isSetMask的支持情况。

针对Tensor前n个数计算和Tensor高维切分计算两种接口，Mask设置方式分为以下情形：

- isSetMask参数情况：

    - 模板参数有isSetMask，支持[接口外设置Mask](../simd_api/basic_api/memory_vector_compute/simd_compute/mask.md#接口外设置Mask)。

    - 模板参数无isSetMask，不支持接口外设置Mask。

    - 模板参数有isSetMask，该参数不起作用。

- Mask设置方式：

    - **接口内已设置Mask**：接口内部调用[SetMaskCount](../simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_count.md)、[SetVectorMask](../simd_api/basic_api/memory_vector_compute/mask_operations/set_vector_mask.md)、[SetMaskNorm](../simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_norm.md)、[ResetMask](../simd_api/basic_api/memory_vector_compute/mask_operations/reset_mask.md)。

    - **接口内通过模板参数控制**：接口内通过isSetMask模板参数控制如何设置Mask。isSetMask为true时表示在接口内设置Mask；isSetMask为false时表示在接口外设置Mask，开发者需使用[SetVectorMask](../simd_api/basic_api/memory_vector_compute/mask_operations/set_vector_mask.md)接口设置Mask值。

    - **接口不支持Mask或者无需设置Mask**。

更多Mask信息可参考[基于全局掩码复用的计算性能优化](../../guides/operator_practice/simd_operator_optimization/vector_compute/mask_reuse_optimization.md)。

注：表中的`-`表示不存在该类型接口。

**表 1**  各接口模板参数isSetMask支持情况汇总表

| 接口类别 | 接口名称 | Tensor前n个数计算 | Tensor高维切分计算 | 
| --- | --- | --- | --- |
| 基础算术 | [Add](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/add.md)、[Exp](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/exp.md)、[Ln](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/ln.md)、[Abs](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/abs.md)、[Reciprocal](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/reciprocal.md)、[Sqrt](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/sqrt.md)、[Rsqrt](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/rsqrt.md)、[Relu](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/relu.md)、[Sub](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/sub.md)、[Mul](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/mul.md)、[Div](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/div.md)、[Max](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/max.md)、[Min](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/min.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 基础算术 | [BilinearInterpolation](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/bilinear_interpolation_isasi.md) | - | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 |
| 基础算术 | [Adds](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/adds.md)、[Muls](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/muls.md)、[Maxs](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/maxs.md)、[Mins](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/mins.md)、[LeakyRelu](../simd_api/basic_api/memory_vector_compute/basic_arithmetic/leaky_relu.md) | &bull; 有isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 逻辑计算 | [Not](../simd_api/basic_api/memory_vector_compute/logical_compute/not.md)、[And](../simd_api/basic_api/memory_vector_compute/logical_compute/and.md)、[Or](../simd_api/basic_api/memory_vector_compute/logical_compute/or.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 逻辑计算 | [ShiftLeft](../simd_api/basic_api/memory_vector_compute/logical_compute/shift_left.md)、[ShiftRight](../simd_api/basic_api/memory_vector_compute/logical_compute/shift_right.md) | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 复合计算 | [Axpy](../simd_api/basic_api/memory_vector_compute/composite_compute/axpy.md)、[MulAddDst](../simd_api/basic_api/memory_vector_compute/composite_compute/mul_add_dst.md)、[MulCast](../simd_api/basic_api/memory_vector_compute/composite_compute/mul_cast.md)、[AddRelu](../simd_api/basic_api/memory_vector_compute/composite_compute/add_relu.md)、[AddReluCast](../simd_api/basic_api/memory_vector_compute/composite_compute/add_relu_cast.md)、[SubRelu](../simd_api/basic_api/memory_vector_compute/composite_compute/sub_relu.md)、[SubReluCast](../simd_api/basic_api/memory_vector_compute/composite_compute/sub_relu_cast.md)、[FusedMulAdd](../simd_api/basic_api/memory_vector_compute/composite_compute/fused_mul_add.md)、[MulAddRelu](../simd_api/basic_api/memory_vector_compute/composite_compute/mul_add_relu.md)、[CastDequant](../simd_api/basic_api/memory_vector_compute/composite_compute/cast_dequant.md)、[AddDeqRelu](../simd_api/basic_api/memory_vector_compute/composite_compute/add_deq_relu.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 比较与选择 | [Compare](../simd_api/basic_api/memory_vector_compute/compare_and_select/compare.md) | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 | &bull; 有isSetMask，不起作用。<br>&bull; 接口无需设置Mask。 |
| 比较与选择 | [Compare（结果存入寄存器）](../simd_api/basic_api/memory_vector_compute/compare_and_select/compare_store_to_register.md) | - | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 比较与选择 | [Compares](../simd_api/basic_api/memory_vector_compute/compare_and_select/compares.md) | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 | &bull; 有isSetMask。<br>&bull; 接口无需设置Mask。 |
| 比较与选择 | [Select](../simd_api/basic_api/memory_vector_compute/compare_and_select/select.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask，不起作用。<br>&bull; 如需使用在接口外部设置Mask的功能，可以调用[不传入mask参数的接口](../simd_api/basic_api/memory_vector_compute/compare_and_select/select.md#函数原型)来实现。 |
| 比较与选择 | [GatherMask](../simd_api/basic_api/memory_vector_compute/compare_and_select/gather_mask.md) | - | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 |
| 类型转换 | [Cast](../simd_api/basic_api/memory_vector_compute/type_conversion/cast.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 归约计算 | [ReduceDataBlock](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_data_block.md)、[ReduceRepeat](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_repeat.md)、[ReducePairElem](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_pair_elem.md) | - | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 归约计算 | [ReduceSum](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_sum.md) | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 |
| 归约计算 | [ReduceMax](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_max.md)、[ReduceMin](../simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_min.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 |
| 数据排布转换 | [Transpose](../simd_api/basic_api/memory_vector_compute/data_layout_conversion/transpose.md)、[TransDataTo5HD](../simd_api/basic_api/memory_vector_compute/data_layout_conversion/trans_data_to5_hd.md) | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 |
| 数据填充 | [Brcb](../simd_api/basic_api/memory_vector_compute/data_padding/brcb.md) | - | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 |
| 数据填充 | [Duplicate](../simd_api/basic_api/memory_vector_compute/data_padding/duplicate.md) | &bull; 无isSetMask。<br>&bull; 接口内已设置Mask。 | &bull; 有isSetMask。<br>&bull; 接口内通过模板参数控制。 |
| 数据填充 | [CreateVecIndex](../simd_api/basic_api/memory_vector_compute/data_padding/create_vec_index.md) | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 |
| 排序组合（ISASI） | [Sort32](../simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/sort32.md)、[MrgSort](../simd_api/basic_api/memory_vector_compute/sort_and_merge_isasi/mrg_sort.md) | - | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 |
| 离散与聚合 | [Gather](../simd_api/basic_api/memory_vector_compute/scatter_gather/gather.md)、[Gatherb](../simd_api/basic_api/memory_vector_compute/scatter_gather/gatherb_isasi.md) | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 | &bull; 无isSetMask。<br>&bull; 接口不支持Mask。 |
| 掩码操作 | [SetMaskCount](../simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_count.md)、[SetMaskNorm](../simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_norm.md)、[SetVectorMask](../simd_api/basic_api/memory_vector_compute/mask_operations/set_vector_mask.md)、[ResetMask](../simd_api/basic_api/memory_vector_compute/mask_operations/reset_mask.md) | - | - |


