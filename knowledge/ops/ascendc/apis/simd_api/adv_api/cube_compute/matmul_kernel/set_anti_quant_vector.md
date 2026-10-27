---
type: API
title: SetAntiQuantVector
description: 在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供一个量化参数向量，该向量的shape为[1,
  N]，N值为Matmul矩阵计算时M/N/K中的N值。对B矩阵的每一列都采用该向量中对应列的伪量化系数进行伪量化。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetAntiQuantVector.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- 310p
source_headers:
- include/adv_api/matmul/matmul.h
---

# SetAntiQuantVector

## 功能说明

在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供一个量化参数向量，该向量的shape为\[1, N\]，N值为Matmul矩阵计算时M/N/K中的N值。对B矩阵的每一列都采用该向量中对应列的伪量化系数进行伪量化。

请在[Iterate](iterate.md)或者[IterateAll](iterate_all.md)之前调用该接口。

## 函数原型

```
__aicore__ inline void SetAntiQuantVector(const LocalTensor<SrcT>& offsetTensor, const LocalTensor<SrcT>& scaleTensor)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offsetTensor | 输入 | 伪量化运算时的参数向量，用于加。SrcT为A_TYPE中对应的数据类型。 |
| scaleTensor | 输入 | 伪量化运算时的参数向量，用于乘。SrcT为A_TYPE中对应的数据类型。 |

## 返回值说明

无

## 约束说明

无
