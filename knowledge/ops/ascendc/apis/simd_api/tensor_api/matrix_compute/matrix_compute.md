---
type: API
title: 矩阵计算（Tensor API）
description: 矩阵计算由数据搬入、mmad计算和结果搬出组成。各阶段通过Tensor的存储位置和Layout选择对应Arch通路，通过Atom配置Trait及运行时Params。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/matrix_compute.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms: []
---

# 矩阵计算（Tensor API）

矩阵计算由数据搬入、`mmad`计算和结果搬出组成。各阶段通过Tensor的存储位置和Layout选择对应Arch通路，通过Atom配置Trait及运行时Params。

- **[概述](overview.md)**
- **[矩阵计算分形介绍](index.md)**
- **[矩阵计算的搬入](cube_compute_load/index.md)**
- **[mmad计算](mmad_compute/index.md)**
- **[mmad计算关键特性说明](mmad_compute_key_features/mmad_compute_key_features.md)**
- **[mmad计算辅助配置接口](mmad_compute_aux_config.md)**
- **[矩阵计算的搬出](cube_compute_store/index.md)**
- **[矩阵搬出关键特性说明](cube_store_key_features/index.md)**

Operation、Trait和Atom的通用组合方式参见[Algorithm](../algorithm.md)。
