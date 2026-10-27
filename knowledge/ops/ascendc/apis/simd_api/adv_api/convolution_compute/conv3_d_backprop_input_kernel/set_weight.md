---
type: API
title: SetWeight
description: 设置卷积反向计算的输入矩阵Weight。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Kernel/SetWeight.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetWeight

## 功能说明

设置卷积反向计算的输入矩阵Weight。

## 函数原型

```
__aicore__ inline void SetWeight(const AscendC::GlobalTensor<SrcT>& weight)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| weight | 输入 | Weight矩阵在Global Memory上的首地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。SrcT表示Weight矩阵的数据类型，当前支持的数据类型为：half、bfloat16_t。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
const Conv3DBackpropInputTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
// ...设置其它参数
gradInput_.SetWeight(weightGm_[offsetB_]);
```
