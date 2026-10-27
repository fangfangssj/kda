---
type: Guide
title: UnitFlag：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。unitFlag的核心功能体现为：Mmad和Copy接口引入了单元标志（unit-flag）机制，通过以内存块为粒度实现精细化的数据同步，从而有效降低同步延迟，提升系统整体性能。当UnitFlag开关打开后，对于L0C
  Buffer中的每个内存块（512B），提供一个单元标志位，用于指示该块是否可读或可写。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_tensor_api/mmad_compute_key_features/unit_flag.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/fundamentals/tensor-api-reference.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/mmad_compute_key_features/unit_flag.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute_tensor_api/cube_store_key_features/f32_channel_split.md
  role: constraint
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- Tensor API UnitFlag
- MmadParams unitFlag
---


# UnitFlag

UnitFlag 是 Mmad 计算与 Copy/Fixpipe 搬出之间的细粒度数据同步机制。它以 L0C Buffer 的 512B 内存块为粒度记录可读/可写状态，不是 isKFull/isNFull 一类排布、满载或对齐参数。

配置入口必须按具体 Tensor API 版本核对。本文保存来源明确的 MmadParams/FixpipeParams 字段和同步含义，不将其他版本 MmadTiling 的描述当作同义声明。

## Tensor API 参数与调用边界

固定 reference 所述 DAV-3510/CANN 9.0.0-beta.2 Tensor API 中，AscendC::Te::MmadParams 和 FixpipeParams 都有 `uint8_t unitFlag=0`；来源把终结阶段使用的 FINAL_ACCUMULATION 记为 3。该数字属于配套的计算/搬出协议，不代表任意调用都应设为 3，也不是 m/n/k 的形状标志。

使用分 K 累加时，应让生产者和消费者对同一 L0C 区域采用匹配的开启状态和终结时机。基础 [UnitFlag](../../apis/simd_api/basic_api/cube_compute_isasi/mmad_compute_key_features/unit_flag.md) 卡给出的协议是：2 开启等待但保持状态；3 在 Mmad 写后把块置为可读、在 Fixpipe 读后恢复可写。把每个中间 K 块都提前发布为最终结果，可能让搬出读到不完整累加。

不要从 unitFlag=3 推导已建立 L0C ping-pong 或所有跨流水同步。它与 cmatrixInitVal 的初值控制不同；[F32 ChannelSplit](../../apis/simd_api/tensor_api/matrix_compute/cube_store_key_features/f32_channel_split.md) 还规定不能同时使能 unitFlag。方向、计算与搬出区域、容量和生命周期需符合对应 SDK 合同；来源未固定上游 master revision，不能直接推广到其他架构和版本。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [unit_flag](../../apis/simd_api/tensor_api/matrix_compute/mmad_compute_key_features/unit_flag.md)
