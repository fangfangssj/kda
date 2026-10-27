---
type: API
title: SetTraverse
description: 设置固定的Matmul计算方向，M轴优先还是N轴优先。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetTraverse.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/matmul/matmul_tiling_base.h
---

# SetTraverse

## 功能说明

设置固定的Matmul计算方向，M轴优先还是N轴优先。

## 函数原型

```
int32_t SetTraverse(MatrixTraverse traverse)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| traverse | 输入 | 设置固定的Matmul计算方向。可选值：MatrixTraverse::FIRSTM/MatrixTraverse::FIRSTN。<br><br>FIRSTM代表先往M轴方向偏移再往N轴方向偏移。<br><br>FIRSTN代表先往N轴方向偏移再往M轴方向偏移。 |

## 返回值说明

-1表示设置失败；0表示设置成功。

## 约束说明

无

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);
tiling.SetTraverse(MatrixTraverse::FIRSTM); // 设置遍历方式
```
