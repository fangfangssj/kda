---
type: API
title: SetInput
description: 设置特征矩阵Input。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/SetInput.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetInput

## 功能说明

设置特征矩阵Input。

## 函数原型

```
__aicore__ inline void SetInput(const AscendC::GlobalTensor<InputT>& input)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| input | 输入 | Input在Global Memory上的首地址。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。特征矩阵Input支持的数据类型为：half、bfloat16_t。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
GlobalTensor<half> inputGm;
inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
conv3dApi.SetInput(inputGm);
```
