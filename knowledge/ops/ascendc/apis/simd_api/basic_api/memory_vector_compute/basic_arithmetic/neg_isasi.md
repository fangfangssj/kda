---
type: API
title: Neg
description: 单目矢量计算接口，对输入tensor按元素取相反数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Neg_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_vec_binary_scalar_intf.h
---

# Neg\(ISASI\)<a name="ZH-CN_TOPIC_0000002193851733"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_binary_scalar_intf.h"`。

Neg属于单目矢量类计算接口，负责将输入的tensor按元素取相反数。计算公式如下：

$dst_i = -src_i$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void Neg(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
```

## 参数说明<a name="section176711403104"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
| ---- | ---- |
| T | 操作数数据类型。 |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| dst | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN、VECCALC、VECOUT。<br>地址对齐约束参考[地址对齐约束](../../../general_description_and_constraints.md)。 |
| src | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN、VECCALC、VECOUT。<br>地址对齐约束参考[地址对齐约束](../../../general_description_and_constraints.md)。 |
| count | 输入 | 参与计算的元素个数。关于该参数的具体说明请参考[连续计算](../simd_compute/continuous_compute.md)。 |

## 数据类型


针对Ascend 950PR&950DT系列产品，支持的数据为int8_t、int16_t、half、int32_t、float、int64_t。


## 返回值说明<a name="section14483414194"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md)。

## 调用示例<a name="section176061616102911"></a>

更多样例可参考[LINK](more_examples.md)。

```cpp
AscendC::Neg(dstLocal, srcLocal, 512);
```

结果示例如下（srcLocal和dstLocal均为int64\_t类型）：

```bash
输入数据(srcLocal): [1 2 3 4 ...]
输出数据(dstLocal): [-1 -2 -3 -4 ...]
```
