---
type: API
title: SetAntiQuantScalar
description: 在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供对B矩阵的所有数据采用同一量化系数进行伪量化的功能。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetAntiQuantScalar.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- 310p
source_headers:
- include/adv_api/matmul/matmul.h
---

# SetAntiQuantScalar

## 功能说明

在Matmul计算时支持A矩阵half类型输入，B矩阵int8类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从GM搬出到L1时，会执行伪量化操作，将B矩阵转化为half类型。本节的伪量化接口提供对B矩阵的所有数据采用同一量化系数进行伪量化的功能。

请在[Iterate](iterate.md)或者[IterateAll](iterate_all.md)之前调用该接口。

## 函数原型

```
__aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offsetScalar | 输入 | 伪量化系数，用于加法。SrcT为A_TYPE中对应的数据类型。 |
| scaleScalar | 输入 | 伪量化系数，用于乘法。SrcT为A_TYPE中对应的数据类型。 |

## 返回值说明

无

## 约束说明

无
