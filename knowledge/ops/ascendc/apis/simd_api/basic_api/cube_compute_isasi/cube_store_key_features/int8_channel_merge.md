---
type: API
title: Int8 Channel Merge
description: 矩阵计算的搬出过程中对于目标数据类型被转换为int8_t、uint8_t的情况，NZ输出时该特性自动启用。硬件会将分形矩阵从16×16扩展为16×32。具体规则如下：若输出通道数N是16的偶数倍，则在N方向上，每两个相邻的16×16分形矩阵将合并为一个单一的16×32分形矩阵。若N是16的奇数倍，则通道1至通道（N–16）将被合并，而最后16个通道保持不变，不进行合并。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_key_features/Int8-Channel-Merge.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# Int8 Channel Merge<a name="ZH-CN_TOPIC_0000002538071280"></a>

## 特性说明

矩阵计算的搬出过程中对于目标数据类型被转换为int8\_t、uint8\_t的情况，NZ输出时该特性自动启用。硬件会将分形矩阵从16×16扩展为16×32。具体规则如下：若输出通道数N是16的偶数倍，则在N方向上，每两个相邻的16×16分形矩阵将合并为一个单一的16×32分形矩阵。若N是16的奇数倍，则通道1至通道（N–16）将被合并，而最后16个通道保持不变，不进行合并。

## 特性约束

硬件自动开启，不能配置。

## 使用示例

例如，当目标数据类型为s8，M为32，N为48时，硬件会先将N方向上前两个16×16的分形矩阵合并成一个16×32的矩阵，然后将剩余的16×16分形矩阵直接搬运至L1 Buffer。

**图1** Int8 Channel Merge示意图<a name="zh-cn_topic_0000002547260781_fig1935632714360"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_Int8_Channel_Merge.png)
