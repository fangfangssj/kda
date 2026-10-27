---
type: API
title: asc_set_l0c_copy_params
description: 头文件路径为："c_api/cube_datamove/cube_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_params.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_set_l0c_copy_params

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

DataCopy从L0C Buffer搬运到Global Memory或L1 Buffer过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。


本接口用于Atlas A3系列产品和Atlas A2系列产品对应的[asc_copy_l0c2gm](../cube_compute_store/asc_copy_l0c2gm_arch_2201.md)和[asc_copy_l0c2l1](../cube_compute_store/asc_copy_l0c2l1_arch_2201.md)；矩阵数量和间隔的图示见[NZ2ND](../cube_store_key_features/nz2_nd.md)。


Ascend 950PR&950DT系列产品使用[asc_set_l0c_copy_nz_para](asc_set_l0c_copy_nz_para.md)，参数单位以该接口说明为准。


## 函数原型

```cpp
__aicore__ inline void asc_set_l0c_copy_params(uint16_t nd_num, uint16_t src_nd_stride, uint16_t dst_nd_stride)
```

## 参数说明

**表1** 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| nd_num       | 输入    | 	ND矩阵的数量，取值范围：nd_num∈[1, 65535]。            |
| src_nd_stride | 输入    | 以分形大小为单位的源步长，源相邻nz矩阵的起始地址间的偏移。<br> src_nd_stride∈[1, 512]，单位：fractal_size 1024B。|
| dst_nd_stride | 输入    | 	目的相邻ND矩阵的起始地址间的偏移。单位为元素。 <br> dst_nd_stride∈[1, 65535]。      |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint16_t nd_num = 2;
uint16_t src_nd_stride = 2;
uint16_t dst_nd_stride = 1;
asc_set_l0c_copy_params(nd_num, src_nd_stride, dst_nd_stride);
```
