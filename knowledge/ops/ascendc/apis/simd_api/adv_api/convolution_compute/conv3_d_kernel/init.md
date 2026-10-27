---
type: API
title: 卷积计算Init
description: Init主要用于对Conv3D对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，同时获取用户声明的Pipe对象，完成内存分配。Tiling参数的具体介绍请参考Conv3D
  Tiling。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/Init.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# Init

## 功能说明

Init主要用于对Conv3D对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，同时获取用户声明的Pipe对象，完成内存分配。Tiling参数的具体介绍请参考[Conv3D Tiling](../conv3_d_tiling/index.md)。

## 函数原型

```
__aicore__ inline void Init(const void* __restrict cubeTiling)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cubeTiling | 输入 | Conv3D对象的Tiling参数，Tiling结构体定义请参见[TConv3DApiTiling结构体](../conv3_d_tiling/t_conv3_d_api_tiling_struct.md)。<br><br>Tiling参数可以通过Host侧[GetTiling](../conv3_d_tiling/get_tiling.md)接口获取，并传递到核函数（Kernel）侧使用。 |

## 返回值说明

无

## 约束说明

-   调用Init接口前必须先初始化TPipe。
-   Init接口必须在IterateAll和End接口前调用，且只能调用一次Init接口，调用顺序如下。

    ```
    Init(...);
    ...
    IterateAll(...);
    End();
    ```

## 调用示例

```
TPipe pipe;
conv3dApi.Init(&tiling);
```

- [QuantConvolution A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/quant_convolution.md)
