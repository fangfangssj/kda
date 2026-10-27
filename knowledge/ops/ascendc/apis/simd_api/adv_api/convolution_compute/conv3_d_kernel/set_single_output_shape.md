---
type: API
title: SetSingleOutputShape
description: 设置单核上结果矩阵Output的形状。 Conv3D高阶API目前支持M合轴模式的输出方式。在M合轴模式下，Conv3D API内部将Wout和Hout视为同一轴处理，输出时先沿Wout方向输出，完成一整行Wout输出后，再进行下一行的Wout输出。
  图1 M合轴模式示意图 !
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/SetSingleOutputShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
---

# SetSingleOutputShape

## 功能说明

设置单核上结果矩阵Output的形状。

Conv3D高阶API目前支持M合轴模式的输出方式。在M合轴模式下，Conv3D API内部将Wout和Hout视为同一轴处理，输出时先沿Wout方向输出，完成一整行Wout输出后，再进行下一行的Wout输出。

**图1**  M合轴模式示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/m_axis_merge_mode.png "M合轴模式示意图")

## 函数原型

```
__aicore__ inline void SetSingleOutputShape(uint64_t singleCo, uint64_t singleDo, uint64_t singleM)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCo | 输入 | 单核上Output的C维度大小。 |
| singleDo | 输入 | 单核上Output的D维度大小。 |
| singleM | 输入 | 单核上Output的M维度大小，即H维度大小与W维度大小的乘积。 |

## 返回值说明

无

## 约束说明

本接口当前仅支持设置Output的C维度、D维度和M维度（即H轴、W轴合并后的维度），不支持设置原始Output的大小。

## 调用示例

```
conv3dApi.SetSingleOutputShape(singleCoreCout, singleCoreDout, singleCoreM);
```
