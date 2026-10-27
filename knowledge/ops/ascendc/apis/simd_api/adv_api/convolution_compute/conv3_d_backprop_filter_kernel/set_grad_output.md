---
type: API
title: SetGradOutput
description: 设置卷积反向计算的输入矩阵GradOutput。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Kernel/SetGradOutput.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetGradOutput

## 功能说明

设置卷积反向计算的输入矩阵GradOutput。

## 函数原型

```
__aicore__ inline void SetGradOutput(const AscendC::GlobalTensor<SrcT>& gradOutput)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gradOutput | 输入 | GradOutput矩阵在Global Memory上的首地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。SrcT表示GradOutput矩阵的数据类型，当前支持的数据类型为：half、bfloat16_t。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
const Conv3DBackpropFilterTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType> gradWeight_;
gradWeight_.Init(&(tilingData->dwTiling));
gradWeight_.SetInput(inputGm_[offsetB_]);
// 设置gradOutput
gradWeight_.SetGradOutput(gradOutputGm_[offsetA_]);
...
```
