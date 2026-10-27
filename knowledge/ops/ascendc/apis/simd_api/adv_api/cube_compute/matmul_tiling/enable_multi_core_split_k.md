---
type: API
title: EnableMultiCoreSplitK
description: 多核场景，通过该接口开启切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/EnableMultiCoreSplitK.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/matmul/bmm_tiling.h
---

# EnableMultiCoreSplitK

## 功能说明

多核场景，通过该接口开启切K轴。不调用该接口的情况下，默认不切K轴。在GetTiling接口调用前使用。

## 函数原型

```
void EnableMultiCoreSplitK(bool flag)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flag | 输入 | 是否开启切K轴。<br>true：开启切K轴。<br>false：不开启切K轴。 |

## 返回值说明

无

## 约束说明

-   在算子中使用该接口时，获取C矩阵结果时仅支持输出到Global Memory。
-   在算子中使用该接口时，需在核函数（Kernel）侧代码中首次将C矩阵分片的结果写入Global Memory之前，先清零Global Memory，随后在获取C矩阵分片的结果时，再开启AtomicAdd累加。如果不预先清零Global Memory，可能会因为累加Global Memory中原始的无效数据而产生精度问题。
-   在算子中使用该接口时，不支持Bias参与矩阵乘计算。

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);

tiling.EnableMultiCoreSplitK(true); // 开启切K轴
```
