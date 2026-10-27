---
type: API
title: SetWeight
description: 设置权重矩阵Weight。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/SetWeight.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# SetWeight

## 功能说明

设置权重矩阵Weight。

## 函数原型

```
__aicore__ inline void SetWeight(const AscendC::GlobalTensor<WeightT>& weight)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| weight | 输入 | Weight在Global Memory上的地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。权重矩阵Weight支持的数据类型为：half、bfloat16_t。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
GlobalTensor<half> weightGm;
weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight));
conv3dApi.SetWeight(weightGm);
```

- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)
