---
type: API
title: SetBiasType
description: 设置Bias的位置，数据格式，数据类型等信息，这些信息需要和kernel侧的设置保持一致。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetBiasType.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# SetBiasType

## 功能说明

设置Bias的位置，数据格式，数据类型等信息，这些信息需要和核函数（Kernel）侧的设置保持一致。

## 函数原型

```
int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | Bias矩阵所在的buffer位置，可设置为：TPosition::GM, TPosition::VECOUT。 |
| type | 输入 | Bias矩阵的数据格式，可设置为：CubeFormat::ND。 |
| dataType | 输入 | Bias矩阵的数据类型，可设置为：DataType::DT_FLOAT、DataType::DT_FLOAT16、DataType::DT_INT32。<br><br>其中，仅在A、B的数据类型均为int8_t时，Bias的数据类型可以设置为int32_t。 |

## 返回值说明

-1表示设置失败； 0表示设置成功。

## 约束说明

无

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);
tiling.SetBiasType(
    matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT); // 设置Bias矩阵
```
