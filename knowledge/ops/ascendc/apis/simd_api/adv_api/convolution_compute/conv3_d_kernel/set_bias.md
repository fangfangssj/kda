---
type: API
title: SetBias
description: 设置偏置矩阵Bias。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/SetBias.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# SetBias

## 功能说明

设置偏置矩阵Bias。

## 函数原型

```
__aicore__ inline void SetBias(const AscendC::GlobalTensor<BiasT>& bias)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| bias | 输入 | Bias在Global Memory上的地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。偏置矩阵Bias支持的数据类型为：half、bfloat16_t。 |

## 返回值说明

无

## 约束说明

在卷积计算中，如果涉及偏置矩阵Bias，必须调用此接口；若卷积计算不涉及Bias，则不应调用此接口。

## 调用示例

```
GlobalTensor<float> biasGm;
biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bias));
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
```

- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)

## 配套 API 与常见组合

- [TPosition](../../../basic_api/aux_data_structures/t_position.md)——卷积使用 bias 时 tiling 侧必须 SetBiasType(TPosition::GM,...) 声明位置（conv2d/conv3d tiling 实证且有 bias 才调用）
