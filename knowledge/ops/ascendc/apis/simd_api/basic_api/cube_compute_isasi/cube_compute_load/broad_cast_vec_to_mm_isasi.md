---
type: API
title: BroadCastVecToMMISASI
description: 将矢量数据广播到矩阵中，每个数据块中的每16个元素会被连续复制16次；当前支持的数据传输通路：Unified Buffer-\>L0C Buffer（VECIN/VECCALC/VECOUT-\>CO1）。
  图1 功能示例 !
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/BroadCastVecToMM_ISASI.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- 310p
---

# BroadCastVecToMM\(ISASI\)<a name="ZH-CN_TOPIC_0000001860143713"></a>

## 功能说明<a name="section618mcpsimp"></a>

将矢量数据广播到矩阵中，每个数据块中的每16个元素会被连续复制16次；当前支持的数据传输通路：UB-\>L0C Buffer（VECIN/VECCALC/VECOUT-\>CO1）。

**图1** 功能示例<a name="fig1730933122314"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/function_example.png "功能示例")

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T, typename U>
__aicore__ inline void BroadCastVecToMM(const LocalTensor<T> &dst, const LocalTensor<U> &src, const int32_t blockCount, const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明<a name="table4835205712588"></a>

| 参数名 | 描述 |
| ---------- | ---------- |
| T | dst的数据类型。 |
| U | src的数据类型。 |

**表2** 参数说明<a name="table284254116544"></a>

| 参数名称 | 类型 | 说明 |
| ---------- | ---------- | ---------- |
| dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor，支持的物理地址为L0C Buffer（TPosition：CO1）。<br>LocalTensor的起始地址需要256个元素对齐。<br>支持的数据类型为：half、int32_t、float。 |
| src | 输入 | 源操作数，输入矢量，类型为LocalTensor，支持的物理地址为UB（TPosition：VECIN/VECCALC/VECOUT）。<br>支持的数据类型需要与dst一致。 |
| blockCount | 输入 | 指定该指令包含的连续广播数据块个数，取值范围：blockCount∈[1, 255]。 |
| blockLen | 输入 | 指定该指令每个连续广播数据块长度，单位为16个元素。取值范围：blockLen∈[1, 255]。 |
| srcGap | 输入 | 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock（32字节）。 |
| dstGap | 输入 | 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为256个元素。 |

## 数据类型

Atlas推理系列产品AI Core，支持的数据类型为：half、int32_t、float。

## 约束说明<a name="section633mcpsimp"></a>

操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

本示例中，输入bias形状为\[1, 32\]，输出c的形状为\[32, 32\]，格式为Nz，调用示例图如下：

**图2** 调用示例图<a name="fig496292825418"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/call_example.png "调用示例图")

本示例仅展示样例中的部分代码。如需运行，请参考[BroadCastVecToMM样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/00_data_movement/broadcast_ub2l0c)实现完整的代码。

```cpp
// brcLocal为TPosition::CO1上的float类型的LocalTensor，srcLocal为TPosition::VECIN上的float类型的LocalTensor
// blockCount = 1，blockLen = 1，连续广播的数据块个数为1，每个数据块包含16个元素，共输出256个元素
// srcGap = 0，dstGap = 1，源操作数与目的操作数之间连续
AscendC::BroadCastVecToMM(brcLocal, srcLocal, 1, 1, 0, 1);
```
