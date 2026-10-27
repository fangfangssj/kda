---
type: API
title: Cube计算(ISASI)总体说明
description: 矩阵计算的搬出是Ascend C编程框架中用于数据搬运的一类核心接口，主要实现L0C Buffer与Global Memory、L1 Buffer、Unified
  Buffer（UB）之间的数据高效传输。该接口系列提供了多种数据搬运模式，能够满足矩阵计算中的数据搬运需求，通常在Mmad接口后配合使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/overall_description.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 总体说明<a name="ZH-CN_TOPIC_0000002538231192"></a>

矩阵计算的搬出是Ascend C编程框架中用于数据搬运的一类核心接口，主要实现L0C Buffer与Global Memory、L1 Buffer、Unified Buffer（UB）之间的数据高效传输。该接口系列提供了多种数据搬运模式，能够满足矩阵计算中的数据搬运需求，通常在Mmad接口后配合使用。

矩阵计算的搬出接口支持多种随路搬运场景，包括**随路量化、随路ReLU、随路格式转换（NZ2ND、NZ2DN）、随路通道拆分（Channel Split）、随路通道合并（Channel Merge）、UB双目标模式**。通过灵活配置不同的参数结构体，开发者可以精确控制数据的搬运路径、起始位置、搬运长度、随路搬运能力等关键信息，充分利用硬件的数据搬运能力，最大化算子的整体执行性能。


矩阵计算的搬出整体流程如下图所示：



**图1** 矩阵计算的搬出整体流程图（[NPU架构版本2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_execution_flow_a2a3.png)



**图2** 矩阵计算的搬出整体流程图（[NPU架构版本3510](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_execution_flow_a5.png)


其中，Fixpipe Buffer不同的量化模式对应的内存排布如下：

**图3** Fixpipe Buffer内存排布图

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_buffer.png)
