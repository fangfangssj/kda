---
type: API
title: SIMD计算概述
description: 矢量计算通过Vector计算单元完成，矢量计算的源操作数和目的操作数均通过Unified Buffer（UB）来进行存储。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/SIMD_compute/overview.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 概述

矢量计算通过Vector计算单元完成，矢量计算的源操作数和目的操作数均通过Unified Buffer（UB）来进行存储。

Memory矢量计算提供[连续计算](continuous_compute.md#连续计算)和[高维切分](high_dim_split.md#高维切分)两种计算模式，并可通过[掩码](mask.md#掩码)实现数据选择性参与计算。

以Add接口为例，两种计算模式的调用方式示意图如下。其中，连续计算简单直观，适合一维Tensor连续数据处理；高维切分计算灵活可控，支持迭代执行和地址间隔配置。开发者可根据实际业务需求选择合适的计算方式。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/multi_level_vector_api.png)
