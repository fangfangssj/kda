---
type: API
title: HcclTilingData
description: HCCL通信Tiling配置结构体，包含Mc2InitTiling和Mc2CcTiling，通过GetTiling接口返回，最多支持8个通信任务。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/TilingData_struct.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# TilingData结构体

## 功能说明

在算子实现中，由Tiling组装通信配置项，核函数（Kernel）实现时将Tiling配置项通过入参传递给核函数（Kernel）侧通信API做通信计算。本节TilingData结构体包括[Mc2InitTiling](#table4835205712588)和[Mc2CcTiling](#table678914014562)，这两个结构体均通过调用[GetTiling](get_tiling.md)接口返回。其中，[Mc2CcTiling](#table678914014562)为具体每个通信任务的参数配置，当算子中有多个通信任务时，可定义多个[Mc2CcTiling](#table678914014562)参数（最多支持定义8个）。

## 参数说明

**表1**  Mc2InitTiling参数说明

<a name="table4835205712588"></a>

| 参数名 | 描述 |
| --- | --- |
| reserved | 初始化通信任务配置。uint8_t*类型，支持最大长度64字节，该结构体仅支持通过接口[GetTiling](get_tiling.md)获取。 |

**表2**  Mc2CcTiling参数说明

<a name="table678914014562"></a>

| 参数名 | 描述 |
| --- | --- |
| reserved | 各通信域中每个通信任务的参数配置。uint8_t*类型，支持最大长度280字节，该结构体仅支持通过接口[GetTiling](get_tiling.md)获取。注意，最多支持配置8个通信任务。 |
