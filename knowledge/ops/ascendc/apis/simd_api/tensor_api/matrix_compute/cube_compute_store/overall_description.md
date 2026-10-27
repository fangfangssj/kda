---
type: API
title: 矩阵计算的搬出总体说明
description: 矩阵计算的搬出接口主要实现L0C Buffer与Global Memory、Unified Buffer（UB）之间的数据高效传输。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/cube_compute_store/overall_description.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 矩阵计算的搬出总体说明

矩阵计算的搬出接口主要实现L0C Buffer与Global Memory、Unified Buffer（UB）之间的数据高效传输。

针对Ascend 950PR&950DT系列产品：

**图1**  矩阵计算搬出整体流程图

![矩阵计算搬出整体流程图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe-20.png)

**表1**  数据通路与存储层级

|源位置|源地址对齐要求|目的位置|目的地址对齐要求|格式转换|典型应用场景|
|--------|--------|--------|--------|--------|--------|
|L0C Buffer|64字节|Global Memory|1字节/32字节|NZ2ND、NZ2DN、NZ2NZ|从L0C Buffer搬出数据至Global Memory。|
|L0C Buffer|64字节|UB|32字节|NZ2ND、NZ2DN、NZ2NZ|从L0C Buffer搬出数据至UB。|
