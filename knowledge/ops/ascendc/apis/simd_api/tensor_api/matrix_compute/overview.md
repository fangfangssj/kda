---
type: API
title: 概述
description: 矩阵计算单元的详细描述请参考LINK。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/overview.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 概述

## 矩阵计算单元

矩阵计算单元的详细描述请参考[LINK](../../basic_api/cube_compute_isasi/overview/cube_compute_unit.md)。

## 矩阵计算流程

Tensor API针对矩阵计算编程模型提供了两类接口，分别承载Cube核中各个通路的搬运能力和计算能力，如下图所示：

**图1**  普通矩阵基础计算流程图<a name="zh-cn_topic_0000002535567224_fig135639216483"></a>

![普通矩阵基础计算流程图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/matrix_computation_workflow_tensor_api.png)

1. 通过`copy`接口将A、B原始矩阵的Global Memory数据搬运到L1 Buffer中（如果存在Bias/随路量化，则同样通过`copy`搬运到L1 Buffer中）。详细内容请参考[矩阵计算的搬入](cube_compute_load/index.md)。

2. 通过`copy`接口将A、B矩阵分别加载到L0A Buffer和L0B Buffer上准备计算（如果存在Bias/随路量化，则通过`copy`将L1 Buffer中的Bias数据/量化系数数据搬运到BiasTable Buffer/Fixpipe Buffer上）。详细内容请参考[矩阵计算的搬入](cube_compute_load/index.md)。

3. 通过`mmad`接口对L0A Buffer、L0B Buffer、BiasTable Buffer上面的数据进行矩阵计算，并输出结果到L0C Buffer上。详细内容请参考[mmad计算](mmad_compute/index.md)。

4. 通过`copy`接口将L0C Buffer的数据进行处理并搬出到Global Memory，`copy`接口可以利用Fixpipe Buffer数据进行如随路量化、Relu等操作。详细内容请参考[矩阵计算的搬出](cube_compute_store/index.md)。
