---
type: API
title: Int8 Channel Merge
description: Channel Merge是矩阵计算结果搬出过程中的硬件自动合并能力，仅在目的格式NZ的输出场景生效，用户无需也不能通过参数手动配置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/cube_store_key_features/int8_channel_merge.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# Int8 Channel Merge

Channel Merge是矩阵计算结果搬出过程中的硬件自动合并能力，仅在目的格式NZ的输出场景生效，用户无需也不能通过参数手动配置。

Int8 Channel Merge：当目标数据类型为`int8_t`或`uint8_t`时，NZ输出自动启用该特性。硬件会将分形矩阵从16×16扩展为16×32。若输出通道数N是16的偶数倍，则N方向上每两个相邻的16×16分形矩阵合并为一个16×32分形矩阵。若N是16的奇数倍，则前N - 16个通道参与合并，最后16个通道保持16×16分形矩阵不变。针对Ascend 950PR&950DT系列产品，`hifloat8_t`、`fp8_e4m3fn_t`类型也支持该特性。

## 使用示例

Int8 Channel Merge：当目标数据类型为`int8_t`，M为32，N为48时，硬件会将N方向上前两个16×16分形矩阵合并成一个16×32分形矩阵，剩余的一个16×16分形矩阵保持不变并输出。

**图1**  Int8 Channel Merge示意图

![Int8 Channel Merge示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_Int8_Channel_Merge.png)

## 约束说明

- Channel Merge为硬件自动使能，不能通过参数配置。
- Int8 Channel Merge在NZ输出且目的类型为`int8_t`或`uint8_t`时自动启用。Ascend 950PR&950DT系列产品上`hifloat8_t`、`fp8_e4m3fn_t` NZ输出也支持该特性。
