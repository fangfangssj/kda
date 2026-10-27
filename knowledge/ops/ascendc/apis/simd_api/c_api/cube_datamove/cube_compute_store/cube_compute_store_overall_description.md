---
type: API
title: 矩阵搬出总体说明
description: 矩阵计算的搬出是Ascend C编程框架中用于数据搬运的一类核心接口，主要实现从L0C Buffer到Global Memory和L1 Buffer的数据传输。该接口系列提供了多种数据搬运模式，能够满足矩阵计算中的数据搬运需求，通常在asc_mmad接口后配合使用。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_compute_store/cube_compute_store_overall_description.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 矩阵搬出总体说明

矩阵计算的搬出是Ascend C编程框架中用于数据搬运的一类核心接口，主要实现从L0C Buffer到Global Memory和L1 Buffer的数据传输。该接口系列提供了多种数据搬运模式，能够满足矩阵计算中的数据搬运需求，通常在asc_mmad接口后配合使用。

矩阵计算的搬出接口支持多种随路搬运场景，包括**随路量化、随路ReLU、随路格式转换（Nz2ND、Nz2DN）、随路通道拆分（Channel Split）、随路通道合并（Channel Merge）**。通过配置接口参数与辅助寄存器，开发者可以精确控制数据的搬运路径、起始位置、搬运长度、随路搬运能力等关键信息，充分利用硬件的数据搬运能力，最大化算子的整体执行性能。


矩阵计算的搬出整体流程如下图所示：



**图1** 矩阵计算的搬出整体流程图（[NPU架构版本2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_execution_flow_a2a3.png)



**图2** 矩阵计算的搬出整体流程图（[NPU架构版本3510](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_execution_flow_a5.png)


其中，Fixpipe Buffer不同的量化模式对应的内存排布如下：

**图3** Fixpipe Buffer内存排布图

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_buffer.png)

搬出接口及架构版本见[L0C到GM](index.md)和[L0C到L1](index.md)。输出布局参见[NZ2NZ](../cube_store_key_features/nz2_nz.md)及[NZ2ND](../cube_store_key_features/nz2_nd.md)；数值处理参见[随路量化](../cube_store_key_features/accompanying_quantization.md)、[随路ReLU](../cube_store_key_features/accompanying_relu.md)及[场景组合](../cube_store_key_features/accompanying_quantization_and_relu_scenario_combination.md)。量化/激活参数搬入见[asc_copy_l12fb](../cube_compute_load/index.md)，配套寄存器配置见[矩阵搬出辅助配置接口](../cube_store_aux_config/index.md)。

Ascend 950PR&950DT系列产品还支持[L0C到UB](asc_copy_l0c2ub.md)搬出；L0C到GM的输出布局还可选择[NZ2DN](../cube_store_key_features/nz2_dn.md)。

