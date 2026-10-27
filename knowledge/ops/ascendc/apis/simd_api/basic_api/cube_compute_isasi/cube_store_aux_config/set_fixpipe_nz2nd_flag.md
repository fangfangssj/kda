---
type: API
title: SetFixpipeNz2ndFlag
description: DataCopy数据搬运（L0C Buffer->GM）过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_aux_config/SetFixpipeNz2ndFlag.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_fixpipe_intf.h
---

# SetFixpipeNz2ndFlag<a id="ZH-CN_TOPIC_0000002569070993"></a>

## 功能说明<a id="zh-cn_topic_0000002516209232_section618mcpsimp"></a>

DataCopy数据搬运（[L0C Buffer->GM](../cube_compute_store/fixpipe_l0_c_to_gm.md)）过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。

## 函数原型<a id="zh-cn_topic_0000002516209232_section620mcpsimp"></a>

```cpp
__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
```


```cpp
// 如下原型仅Ascend 950PR&950DT系列产品支持
__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint32_t dstNdStride)
```


## 参数说明<a id="zh-cn_topic_0000002516209232_section622mcpsimp"></a>


针对如下产品型号，参数说明见下表：

Atlas A3系列产品


Atlas A2系列产品


Atlas 200I/500 A2推理产品

**表1** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ---------- | ---------- | ---------- |
| ndNum | 输入 | nd矩阵的数量，取值范围∈[1, 65535]。（[具体含义图示参见NZ2ND介绍特性章节](../cube_store_key_features/nz2_nd.md)） |
| srcNdStride | 输入 | 以分形大小为单位的源步长，源相邻nz矩阵的偏移（头与头）。当ndNum大于1时，取值范围∈[1, 512]；当ndNum配置为1时，srcNdStride可配置为0，此时不生效。单位为1024B。 |
| dstNdStride | 输入 | 目的相邻nd矩阵的偏移（头与头）。取值范围dstNdStride∈[1, 65535]，单位为element元素。 |




针对Ascend 950PR&950DT系列产品：

**表2** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ---------- | ---------- | ---------- |
| ndNum | 输入 | nd矩阵的数量，取值范围∈[1, 65535]。（[具体含义图示参见NZ2ND介绍特性章节](../cube_store_key_features/nz2_nd.md)） |
| srcNdStride | 输入 | 以分形大小为单位的源步长，源相邻nz矩阵的偏移（头与头）。当ndNum配置为1时，srcNdStride配置为0即可，不生效。取值范围∈[0, 65535]，单位为C0_SIZE。 |
| dstNdStride | 输入 | 目的相邻nd矩阵的偏移（头与头）。取值范围dstNdStride∈[1, 2^32-1]，单位为element元素。 |




针对如下产品型号，参数说明见下表：

Kirin X90处理器系列产品 


Kirin 9030处理器系列产品 

**表3** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ---------- | ---------- | ---------- |
| ndNum | 输入 | nd矩阵的数量，取值范围∈[1, 65535]。（[具体含义图示参见NZ2ND介绍特性章节](../cube_store_key_features/nz2_nd.md)） |
| srcNdStride | 输入 | 以分形大小为单位的源步长，源相邻nz矩阵的偏移（头与头）。当ndNum配置为1时，srcNdStride配置为0即可，不生效。取值范围∈[0, 65535]，单位为C0_SIZE。 |
| dstNdStride | 输入 | 目的相邻nd矩阵的偏移（头与头）。取值范围dstNdStride∈[0, 65535]，单位为element元素。 |



## 数据类型

参数的数据类型见[函数原型](#zh-cn_topic_0000002516209232_section620mcpsimp)。

## 返回值说明<a id="section640mcpsimp"></a>

无

## 约束说明<a id="zh-cn_topic_0000002516209232_section633mcpsimp"></a>

- 调用DataCopy接口并且需要进行随路NZ格式转换为ND格式之前，必须先调用SetFixpipeNz2ndFlag设置格式转换配置。

## 调用示例<a id="section6461234123118"></a>

完整示例可参考[data_copy_l0c2gm](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/00_data_movement/data_copy_l0c2gm)。

```cpp
uint16_t ndNum = 2;
uint16_t srcNdStride = 2;
uint16_t dstNdStride = 1;
AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride); // 设置FIX搬运NZ格式到ND格式转换的参数
```

- [QuantBatchMatmulInplaceAdd 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/quant_batch_matmul_inplace_add_950.md)
- [grouped_matmul_swiglu_quant 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md)
- [AllGatherMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/all_gather_matmul.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul.md)
- [MatmulReduceScatter A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_reduce_scatter.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
