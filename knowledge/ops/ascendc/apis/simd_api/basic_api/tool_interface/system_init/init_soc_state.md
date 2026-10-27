---
type: API
title: InitSocState
description: 初始化AI Core全局状态，包括AIC与AIV的公共状态及各自特有状态；不同产品实际执行的初始化项存在差异。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_init/InitSocState.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# InitSocState

## 功能说明

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

本接口对AI Core的全局状态进行初始化，包括AIC（Cube Core）与AIV（Vector Core）的公共状态及各自特有状态。由于不同产品的实现存在差异，实际执行的初始化项也有所不同。

**接口涉及的全部初始化操作汇总如下：**

| 状态类别 | 初始化内容 |
| ---- | ---- |
| 公共状态 | 初始化[原子累加状态](../../atomic_operations/disable_dma_atomic.md)，关闭数据搬运随路原子操作功能。 |
| 公共状态 | 初始化[Mask工作模式](../../memory_vector_compute/simd_compute/mask.md#mask-mode)为Normal模式。 |
| 公共状态 | 初始化[用于AddDeqRelu/CastDequant/Cast的s322fp16场景的scale量化参数](../../memory_vector_compute/type_conversion_aux_config/set_deq_scale.md#func1)为1。 |
| 公共状态 | 初始化[CTRL寄存器](../../special_register_access/set_ctrl_spr_isasi.md#tab-950)中除`CTRL[48]`以外的比特位为默认值。 |
| 公共状态 | 初始化[原子操作开启位与原子操作类型](../../atomic_operations/set_store_atomic_config_isasi.md)，为无效的原子操作类型。 |
| AIC | 初始化[LoadData（卷积数据搬运）接口调用时Pad填充的数值](../../cube_compute_isasi/cube_load_aux_config/set_load_data_padding_value.md#功能说明)为0。 |
| AIC | 初始化[使用LoadData（卷积数据搬运）时L1 Buffer（A1/B1）的边界值](../../cube_compute_isasi/cube_load_aux_config/set_load_data_boundary.md#功能说明)为0，表示无边界。 |
| AIV | 将[Mask](../../memory_vector_compute/simd_compute/mask.md)配置为全1，表示所有数都参与计算。 |
| AIV | [重置loop mode的参数](../../memory_vector_compute/data_move_aux_config/reset_loop_mode_para.md)。 |

**各产品实际执行的初始化项如下：**


**表1** 针对Ascend 950PR&950DT系列产品本接口执行的初始化项

| 状态类别 | 初始化内容 |
| ---- | ---- |
| 公共状态 | 原子累加状态 |
| 公共状态 | Mask工作模式 |
| 公共状态 | 用于AddDeqRelu/CastDequant/Cast的s322fp16场景的scale量化参数 |
| 公共状态 | CTRL寄存器 |
| 公共状态 | 原子操作开启位与原子操作类型 |
| AIC | LoadData（卷积数据搬运）接口调用时Pad填充的数值 |
| AIV | Mask |
| AIV | loop mode的参数 |




**表2** 针对Atlas A3系列产品本接口执行的初始化项

| 状态类别 | 初始化内容 |
| ---- | ---- |
| 公共状态 | 原子累加状态 |
| 公共状态 | Mask工作模式 |
| AIC | LoadData（卷积数据搬运）接口调用时Pad填充的数值 |
| AIC | 使用LoadData（卷积数据搬运）时L1 Buffer（A1/B1）的边界值 |
| AIV | Mask |




**表3** 针对Atlas A2系列产品本接口执行的初始化项

| 状态类别 | 初始化内容 |
| ---- | ---- |
| 公共状态 | 原子累加状态 |
| 公共状态 | Mask工作模式 |
| AIC | LoadData（卷积数据搬运）接口调用时Pad填充的数值 |
| AIC | 使用LoadData（卷积数据搬运）时L1 Buffer（A1/B1）的边界值 |
| AIV | Mask |




**表4** 针对Atlas推理系列产品AI Core和Atlas推理系列产品Vector Core本接口执行的初始化项

| 状态类别 | 初始化内容 |
| ---- | ---- |
| 公共状态 | 原子累加状态 |



## 函数原型

```cpp
__aicore__ inline void InitSocState()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- 在实际运行中，这些值可能被前序执行的算子修改，若不调用该接口进行初始化，非预期的值可能导致计算结果出现精度错误。

  例如前序算子使用Counter模式但未重置为Normal模式，当前算子以默认的Normal模式设置Mask时，会导致Mask设置不符合预期，进而引发精度错误。
- 在[TPipe框架编程](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/tpipe_tque_principles.md)中，初始化过程由TPipe完成，无需开发者关注；在[静态Tensor编程](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)的场景中，用户必须在核函数（Kernel）入口处调用此函数来初始化AI Core状态。

## 调用示例

```cpp
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // 静态Tensor编程方式中需要开发者手动调用InitSocState()接口初始化全局状态寄存器。
    AscendC::InitSocState();
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
```

- [DualLevelQuantMatmulWeightNz 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/dual_level_quant_matmul_weight_nz_950.md)
- [GroupedMatmulFinalizeRouting 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_finalize_routing_950.md)
- [RotateQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/rotate_quant_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [TransposeBatchMatMul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_batch_mat_mul_950.md)
- [GroupedMatmulSwigluQuant 950 Direct Invoke 算子知识](../../../../../operators/direct/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [TransposeBatchMatMulWeightNZ 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_batch_mat_mul_weight_nz_950.md)
- [FlatQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/flat_quant_950.md)
- [GRU 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/rnn/gru_950.md)

## 配套 API 与常见组合

- [Alloc](../../resource_management/local_mem_allocator/alloc.md)——LocalMemAllocator/Alloc 即静态Tensor编程范式的分配机制
- [LocalMemAllocatorConstructor](../../resource_management/local_mem_allocator/local_mem_allocator_constructor.md)——构造 LocalMemAllocator 即进入静态Tensor编程，核入口必须 InitSocState，P(b|a)=1.00 全量成立
- [SetFlag/WaitFlag](../../sync_control/intra_core_sync/set_flag_wait_flag_isasi.md)——自管 eventID 的 SetFlag/WaitFlag 手动同步属静态Tensor编程（SetFlag 卡约束明文）
