---
type: API
title: DataCopyL1ToGMContinuous
description: 将矩阵从L1 Buffer连续搬运至Global Memory，数据搬运时格式和内容保持不变。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/DataCopy_L1ToGM_continuous.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# DataCopy（L1ToGM连续数据搬运）<a name="ZH-CN_TOPIC_0000002563687928"></a>

## 功能说明<a name="zh-cn_topic_0000002540198372_section106841136114319"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

该接口能够将矩阵从L1 Buffer连续搬运至Global Memory，数据搬运时格式和内容保持不变。

## 函数原型<a name="zh-cn_topic_0000002540198372_section82039854412"></a>

```cpp
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
```

## 参数说明<a name="zh-cn_topic_0000002540198372_section16128134420472"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
| ---------- | ---------- |
| T | 源操作数或者目的操作数的数据类型。支持的数据类型请参考[数据类型](#zh-cn_topic_0000002540198372_section4219135304818)。 |

**表2** 参数说明

| 参数名称 | 输入/输出 | 含义 |
| ---------- | ---------- | ---------- |
| dst | 输出 | 目的操作数，类型为GlobalTensor，存储位置为Global Memory，地址需要1字节对齐。 |
| src | 输入 | 源操作数，类型为LocalTensor，存储位置为L1 Buffer，地址需要32字节对齐。 |
| count | 输入 | 参与搬运的元素个数。<br>**注：count * sizeof(T)需要32字节对齐，若未对齐，搬运量会向下取整到32字节对齐。** |

以half数据类型为例，源操作数的shape为1 \* 128。当count = 128时，[图1](#zh-cn_topic_0000002540198372_fig826614454819)将源操作数中128个元素连续搬运至目的操作数。

**图1** 连续搬运示意图<a id="zh-cn_topic_0000002540198372_fig826614454819"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/continuous_data_copy_diagram.png)

## 数据类型<a id="zh-cn_topic_0000002540198372_section4219135304818"></a>

源矩阵和目的矩阵支持的数据类型保持一致。


针对Atlas A3系列产品，支持数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。



Atlas A2系列产品，支持数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、double。



针对Kirin X90处理器系列产品，支持数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float、int64_t、uint64_t、double。



Kirin 9030处理器系列产品，支持数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、int32_t、uint32_t、float、int64_t、uint64_t、double。


## 返回值说明

无

## 约束说明<a name="zh-cn_topic_0000002540198372_section2045914466492"></a>

- 位于L1 Buffer的源地址必须32字节对齐，位于Global Memory的目的地址必须1字节对齐。
- 调用连续搬运接口时，count \* sizeof\(T\)需要32字节对齐，若未对齐，则搬运量会向下取整到32字节对齐。

- 针对如下产品型号：
    
    Atlas A3系列产品
    
    
    Atlas A2系列产品
    
    在跨卡通信算子开发场景，DataCopy类接口支持跨卡数据搬运，仅支持HCCS物理链路，不支持其他通路；开发者开发过程中，需要关注涉及卡间通信的物理通路，可通过npu-smi info -t topo命令查询HCCS物理链路。


## 调用示例<a name="zh-cn_topic_0000002540198372_section088124295117"></a>

L1到GM连续数据搬运的样例请参考[L1到GM搬运样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/06_compatibility_guide/data_copy_l1togm)。
