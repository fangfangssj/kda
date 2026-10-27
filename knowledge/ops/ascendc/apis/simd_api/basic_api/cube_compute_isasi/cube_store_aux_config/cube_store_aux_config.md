---
type: API
title: 矩阵搬出寄存器配置说明
description: 本章节涉及寄存器在相同核函数内部会维持状态直到被人为改变，而在不同核函数内部不会继承之前的状态。
tags:
- task.lookup
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_aux_config/cube_store_aux_config.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 矩阵搬出寄存器配置说明<a name="ZH-CN_TOPIC_0000002568950985"></a>

本章节涉及寄存器在相同核函数（Kernel）内部会维持状态直到被人为改变，而在不同核函数（Kernel）内部不会继承之前的状态。

- **[SetFixPipeConfig](set_fix_pipe_config.md)**

- **[SetFixpipePreQuantFlag](set_fixpipe_pre_quant_flag.md)**

- **[SetFixpipeNz2ndFlag](set_fixpipe_nz2nd_flag.md)**

- **[SetFixPipeClipRelu](set_fix_pipe_clip_relu.md)**

- **[SetFixPipeAddr](set_fix_pipe_addr.md)**
