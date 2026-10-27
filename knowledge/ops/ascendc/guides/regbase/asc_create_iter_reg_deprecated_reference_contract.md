---
type: Guide
title: asc_create_iter_reg (废弃)：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。该接口已废弃，请使用asc_update_addr_reg实现此功能。 地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量，该重载接口同步支持1-4个offset参数，并支持初始化b8/b16/b32位宽的寄存器，适配1-4维Tensor。底层在多维循环中自动计算地址偏移，消除标量地址运算开销。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_data_types/asc_create_iter_reg_deprecated.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_create_iter_reg.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_create_iter_reg
- asc_create_iter_reg_b8
- asc_create_iter_reg_b16
- asc_create_iter_reg_b32
---

# asc_create_iter_reg (废弃)

该接口已废弃，请使用asc_update_addr_reg实现此功能。 地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量，该重载接口同步支持1-4个offset参数，并支持初始化b8/b16/b32位宽的寄存器，适配1-4维Tensor。底层在多维循环中自动计算地址偏移，消除标量地址运算开销。
函数原型：iter_reg asc_create_iter_reg_b32(uint32_t offset)
返回值：地址寄存器。
约束：请勿在循环内使用 if/else。；iter_reg必须在最内层循环体内赋值使用。；不同步长的操作数，建议使用多个iter_reg，避免地址被覆盖。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 历史单偏移声明

固定 950PR/950DT reference 给出三个单 offset 创建形式，返回类型为 iter_reg：

```cpp
__simd_callee__ inline iter_reg asc_create_iter_reg_b32(uint32_t offset);
__simd_callee__ inline iter_reg asc_create_iter_reg_b16(uint32_t offset);
__simd_callee__ inline iter_reg asc_create_iter_reg_b8(uint32_t offset);
```

offset 是 uint32_t 输入，用于初始化迭代寻址的地址偏移寄存器；流水为 PIPE_V。固定来源未明确 offset 的物理单位，也未列出多维重载的完整声明，不能从 b8/b16/b32 名称补造字节换算或从现有概述推导更多旧原型。

当前迁移到 [asc_update_addr_reg](../../apis/simd_api/c_api/reg_compute/reg_addr_reg/asc_update_addr_reg.md)，寄存器类型也由 iter_reg 更名为 addr_reg。保留最内层循环赋值、避免循环内 if/else 及不同步长使用多个寄存器的约束；不要将这些历史创建声明当成新接口的逐字签名。

## 相关

- [AscUpdateAddrReg](../../apis/simd_api/c_api/reg_compute/reg_addr_reg/asc_update_addr_reg.md)（`same_topic`）：该接口已废弃，请使用asc_update_addr_reg实现此功能。 地址寄存器通过该接口初始化，然后在循环之中使用地址寄存器存储地址偏移量，该重载接口同步支持1-4个offset参数，并支持初始化b8/b16/b32位宽的寄存器，适配1-4维Tensor。底层在多维循环中自动计算地址偏移，消除标量地址运算开销。；>旧接口 asc_create_iter_reg_ 已废弃，请使用 asc_update_addr_reg_ 替代。旧接口返回类型 iter_reg 也已更名为 addr_reg。。两卡从旧接口和新接口两端明确声明一对一替代，并保留同一多维迭代寻址功能。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_create_iter_reg (废弃)](../../apis/simd_api/c_api/deprecated_interface/asc_create_iter_reg_deprecated.md)
