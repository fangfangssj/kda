---
type: API
title: 矩阵计算单元（概述）
description: Cube计算单元专用于执行矩阵运算，直接访问的专用缓存如下：L0A Buffer用于存储左矩阵，L0B Buffer用于存储右矩阵，L0C
  Buffer则用于存储初始化累加值和矩阵计算结果。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/overview/cube_compute_unit.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 矩阵计算单元<a name="ZH-CN_TOPIC_0000002538071190"></a>

Cube计算单元专用于执行矩阵运算，直接访问的专用缓存如下：L0A Buffer用于存储左矩阵，L0B Buffer用于存储右矩阵，L0C Buffer则用于存储初始化累加值和矩阵计算结果。


下图高亮部分展示了Cube计算单元及其直接访问的专用缓存。



**图1** [NPU架构版本2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)矩阵计算单元架构图<a name="zh-cn_topic_0000002513373312_fig88330137205"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/architecture_of_cube_compute_unit_a2a3.png "矩阵计算单元架构图-A2A3")



**图2** [NPU架构版本3510](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)矩阵计算单元架构图<a name="zh-cn_topic_0000002513373312_fig1082103513519"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/npu_3510_hw_arch_cube.png "矩阵计算单元架构图-950")


矩阵计算关联的存储单元为：

| Buffer类型 | 说明 |
| ----------- | ------ |
| L0A Buffer | AI Core内部物理存储单元，通常用于存储矩阵计算的左矩阵。 |
| L0B Buffer | AI Core内部物理存储单元，通常用于存储矩阵计算的右矩阵。 |
| L0C Buffer | AI Core内部物理存储单元，通常用于存储矩阵计算的结果，以及矩阵累加初始化值。 |
| L1 Buffer | AI Core内部物理存储单元，空间相对较大，通常用于缓存矩阵计算的输入数据。矩阵计算的输入一般需要从GM搬运到L1 Buffer，然后分别搬运到L0A Buffer和L0B Buffer。 |
| Fixpipe Buffer | AI Core内部物理存储单元，通常用于存储Fixpipe搬运过程中所需的量化参数等数据。 |
| BiasTable Buffer | 偏置存储，AI Core内部物理存储单元，通常用于存储矩阵计算所需的Bias（偏置）数据。 |
