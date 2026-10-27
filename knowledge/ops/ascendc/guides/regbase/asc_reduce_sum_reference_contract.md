---
type: Guide
title: AscReduceSum：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据mask对源操作数进行归约求和操作，将所有参与计算的元素之和保存到目的操作数。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reduce_compute/asc_reduce_sum.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_reduce_sum.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_reduce_sum
---

# AscReduceSum

根据mask对源操作数src进行归约求和操作，用于将src中的所有参与计算的元素求和，得到的结果保存在dst中的0位置。
约束：指令内累加顺序采用二叉树累加方式，结果连续写入到目的操作数，目的操作数中的其它元素置0。；当所有元素均不参与计算（mask全为0）时，将0写入目的操作数对应位置（特别的，对于浮点数为+0）。；对于输入为uint16_t/int16_t类型的情况，会提升精度到uint32_t/int32_t进行计算。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 六组 C 输入输出类型合同

固定 950PR/950DT reference 列出的 C 声明如下，src 与 dst 不总是相同类型：

```cpp
__simd_callee__ inline void asc_reduce_sum(vector_int32_t& dst, vector_int16_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_sum(vector_int32_t& dst, vector_int32_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_sum(vector_uint32_t& dst, vector_uint16_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_sum(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_sum(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_reduce_sum(vector_float& dst, vector_float src, vector_bool mask);
```

有符号 int16 输入提升到 int32 目的寄存器，无符号 uint16 输入提升到 uint32；int32、uint32、half、float 则使用所列同类型配对。不能仅凭“支持 16 位输入”创建 half→float 的 C 重载，来源给出的 half 配对为 vector_half→vector_half。mask 表示参与求和的输入元素，返回 void，流水为 PIPE_V。

原卡的 lane0 结果位置、其余元素清零、空 mask 写零及浮点 +0、二叉树累加约束继续适用。旧来源的“mask 未筛选的元素在输出中置零”是泛化参数描述，不能误解为每个输入 lane 在同位置产生一个独立归约结果。验证分别检查归约值、结果位置、无效 lane 不参与、空 mask 和 16→32 位配对；不要将普通求和、逐数据块归约和成对归约当作同一调用合同。

这些声明来自固定文档，未在本卡进行 SDK 编译或 NPU 验证，也不保证超出已列集合的 dtype/累加范围可用。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_reduce_sum](../../apis/simd_api/c_api/reg_compute/reg_reduce/asc_reduce_sum.md)
