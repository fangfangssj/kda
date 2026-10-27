---
type: API
title: Mull
description: 双目矢量计算接口，对前count个数据按元素相乘，结果写入dst0，溢出部分写入dst1。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Mull_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_vec_binary_scalar_intf.h
---

# Mull\(ISASI\)<a name="ZH-CN_TOPIC_0000002193851737"></a>

## 功能说明<a name="section1769835112451"></a>

头文件路径为：`"basic_api/kernel_operator_vec_binary_scalar_intf.h"`。

Mull属于双目矢量类计算接口，对前count个输入数据src0、src1按元素相乘操作，将结果写入dst0，溢出部分写入dst1。计算过程如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/mull.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void Mull(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
| ---- | ---- |
| T | 操作数数据类型。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| dst0、dst1 | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN、VECCALC、VECOUT。<br>地址对齐约束参考[地址对齐约束](../../../general_description_and_constraints.md)。 |
| src0、src1 | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN、VECCALC、VECOUT。<br>地址对齐约束参考[地址对齐约束](../../../general_description_and_constraints.md)。 |
| count | 输入 | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |

## 数据类型


针对Ascend 950PR&950DT系列产品，支持的数据为int32_t、uint32_t。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md)。

## 调用示例<a name="section642mcpsimp"></a>

更多样例可参考[LINK](more_examples.md)。

```cpp
AscendC::Mull(dst0Local, dst1Local, src0Local, src1Local, 512);
```

结果示例如下：

```bash
输入数据src0Local：[-2039520159    95829766 -1522913829 ... ]
输入数据src1Local：[-1791575569  -291976917 -1973597151 ... ]
输出数据dst0Local：[-1560359025   212356098  1219226939 ... ]
输出数据dst1Local：[850752575   -6514620  699800065 ... ]
```
