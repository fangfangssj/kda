---
type: API
title: 随路ReLU
description: 矩阵计算的搬出过程中支持随路ReLU能力，当前支持如下随路ReLU能力。DataCopyCO12DstParams结构体参数定义中reluPre可设置为1开启Normal
  ReLU，结合clipReluPre可进一步开启ClipReLU能力。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_key_features/accompanying_relu.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 随路ReLU<a name="ZH-CN_TOPIC_0000002538231206"></a>

## 特性说明

矩阵计算的搬出过程中支持随路ReLU能力，当前支持如下随路ReLU能力。**DataCopyCO12DstParams结构体参数**定义中reluPre可设置为1开启Normal ReLU，结合clipReluPre可进一步开启ClipReLU能力。

- 参数配置为0时，即不开启随路ReLU能力。

- 参数配置为1时，即Normal ReLU，无需配置额外寄存器，对输出数据执行激活处理：`y = max(0, x)`。

- 开启Normal ReLU（reluPre=1）且同时配置clipReluPre=1并调用SetFixPipeClipRelu接口时，即Normal ReLU + ClipReLU，先执行`y = max(0, x)`，再执行`y = min(clipReluMaxVal, y)`，其中clipReluMaxVal通过SetFixPipeClipRelu接口设置。详见[SetFixPipeClipRelu](../cube_store_aux_config/set_fix_pipe_clip_relu.md)。

与随路量化组合使用的详细信息请参考[随路量化与随路ReLU场景组合](accompanying_quantization_and_relu_scenario_combination.md)。
