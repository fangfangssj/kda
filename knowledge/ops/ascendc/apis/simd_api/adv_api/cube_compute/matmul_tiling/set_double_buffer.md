---
type: API
title: SetDoubleBuffer
description: 设置A/B/C/Bias是否开启double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。
  该接口为预留接口，当前版本暂不支持。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetDoubleBuffer.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/matmul/matmul_tiling_base.h
---

# SetDoubleBuffer

## 功能说明

设置A/B/C/Bias是否开启double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

**该接口为预留接口，当前版本暂不支持。**

## 函数原型

```
int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| a | 输入 | 设置A矩阵是否开启double buffer。 |
| b | 输入 | 设置B矩阵是否开启double buffer。 |
| c | 输入 | 设置C矩阵是否开启double buffer。 |
| bias | 输入 | 设置Bias矩阵是否开启double buffer。 |
| transND2NZ | 输入 | 设置是否需要[ND2NZ](../../../basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md)。 |
| transNZ2ND | 输入 | 设置是否需要[NZ2ND](../../../basic_api/cube_compute_isasi/cube_compute_store/data_copy_l0_c_to_gm.md)。 |

## 返回值说明

-1表示设置失败； 0表示设置成功。

## 约束说明

无
