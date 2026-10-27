---
type: Guide
title: AscUpdateMask：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据元素个数生成mask并自动扣减当前向量处理单元元素数，支持b8、b16、b32位宽模式
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_data_types/asc_update_mask.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_update_mask.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_update_mask
- asc_update_mask_b8
- asc_update_mask_b16
- asc_update_mask_b32
---

# AscUpdateMask

根据调用前的元素个数 value 生成 mask，并按一个向量的元素容量饱和扣减 value；不足一个向量时更新为 0。支持b8、b16、b32三种位宽模式，由于VL=256B，各模式的向量处理单元元素个数为： 图1  asc_update_mask更新流程
函数原型：vector_bool asc_update_mask_b8(uint32_t& value)
返回值：vector_bool，掩码寄存器。
约束：掩码寄存器的数量上限为8，超过上限的掩码寄存器会写入预留的8K UB内存中，可能引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若两者均可用，优先复用寄存器。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 三种 C 声明与计数更新

以下为固定 950PR/950DT reference 中的声明，补上分号用于声明书写；未验证所有 SDK 版本均提供相同接口：

```cpp
__simd_callee__ inline vector_bool asc_update_mask_b8(uint32_t& value);
__simd_callee__ inline vector_bool asc_update_mask_b16(uint32_t& value);
__simd_callee__ inline vector_bool asc_update_mask_b32(uint32_t& value);
```

`value` 是输入输出参数，类型为可变 `uint32_t&`。设本轮调用前值为 n，一个向量容纳该位宽元素数为 VL_T：n≥VL_T 时返回全有效 mask；0≤n<VL_T 时只使逻辑元素 0..n-1 有效，n=0 时无有效元素。生成 mask 使用调用前的 n，随后引用参数变成：

```cpp
value = (value < VL_T) ? 0 : (value - VL_T);
```

不能先无条件执行无符号减法再生成 mask，否则小于 VL_T 的尾块会下溢。按本卡 VL=256B 的前提，b8/b16/b32 分别对应 256/128/64 个元素；这是由 VL 和位宽推得的元素数，不是三个 API 都处理 256 个元素。返回值为 vector_bool，流水为 PIPE_V。

地址偏移应以本轮实际处理范围和稳定 offset 维护；不要让 API 和外层循环重复扣减同一 remaining，也不要用更新后的 value 反推本轮起点。通用原则见[尾块与掩码](valid_lengths_and_precision.md)，C++ 模式及更新原理见 [MaskReg](../../apis/simd_api/basic_api/reg_vector_compute/register_data_types/mask_reg.md)。这不意味着 C++ 的全部模板范围都存在对应 C 函数。

来源 b32、127 元素示意可分为 64+63 两轮：第一次后 value=63，第二次后 value=0。源代码的 `i++>` 是循环语法笔误，不作为可编译样例保留。验证应包含 0、VL_T-1、VL_T、VL_T+1 及多轮尾块，同时检查引用参数与有效 lane；本卡未执行 SDK 编译或 NPU 测试。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_update_mask](../../apis/simd_api/c_api/reg_compute/reg_mask/asc_update_mask.md)
