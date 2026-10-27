---
type: API
title: 卷积计算End
description: 在Conv3D计算完成后，必须调用一次End函数，以清除EventID并释放内部申请的临时内存。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/End.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# End

## 功能说明

在Conv3D计算完成后，必须调用一次End函数，以清除EventID并释放内部申请的临时内存。

## 函数原型

```
void End()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

End接口必须在IterateAll接口后调用。

## 调用示例

```
conv3dApi.End();
```

- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)
