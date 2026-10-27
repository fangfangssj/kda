---
type: API
title: AbsSub
description: 将src0与src1相减再求绝对值，并将计算结果写入dst。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.compute.composite
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/composite_compute/AbsSub_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_vec_binary_intf.h
---

# AbsSub\(ISASI\)<a name="ZH-CN_TOPIC_0000002158325168"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_binary_intf.h"`。

将src0与src1相减再求绝对值，并将计算结果写入dst。计算公式如下：

$$
dst_i = |src0_i - src1_i|
$$

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void AbsSub(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1, const uint32_t count)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| :----- | :--- |
| T | 操作数数据类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------- | :--- |
| dst | 输出 | 目的操作数。<br>类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>LocalTensor的起始地址需要32字节对齐。 |
| src0、src1 | 输入 | 源操作数。<br>类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>LocalTensor的起始地址需要32字节对齐。<br>两个源操作数的数据类型需要与目的操作数保持一致。 |
| count | 输入 | 参与计算的元素个数。 |

## 数据类型


Ascend 950PR&950DT系列产品，T支持的数据类型为：half、float。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 操作数地址对齐要求请参见[通用地址对齐约束](../../../general_description_and_constraints.md#section796754519912)。

- 操作数地址重叠约束请参考[通用地址重叠约束](../../../general_description_and_constraints.md#section668772811100)。


- 该接口通过VF调用[Reg矢量计算API](../../reg_vector_compute/index.md)实现兼容，当参数count取值为0时，不保证该接口将被视为NOP（空操作）。
  

## 调用示例<a name="section642mcpsimp"></a>

```cpp
AscendC::AbsSub(dstLocal, src0Local, src1Local, 512);
```

结果示例如下：

```plain
输入数据src0Local：[1 2 3 ... 512]
输入数据src1Local：[513 514 515 ... 1024]
输出数据dstLocal：[512 512 512 ... 512]
```
