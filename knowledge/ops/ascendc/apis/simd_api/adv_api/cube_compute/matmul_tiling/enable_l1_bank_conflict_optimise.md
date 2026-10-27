---
type: API
title: EnableL1BankConflictOptimise
description: 根据GetTiling接口计算出的Tiling参数，获取是否可以开启L1 Bank冲突优化功能。若可以开启该功能，则与TilingKey机制配合使用，通过增加TilingKey，关联Host侧与Kernel侧实现，并在Kernel侧增加代码实现分支，将MatmulConfig中的enableL1BankConflictOptimise设置为true，即可优化L1上的Bank冲突。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/EnableL1BankConflictOptimise.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# EnableL1BankConflictOptimise

## 功能说明

根据[GetTiling](get_tiling.md)接口计算出的Tiling参数，获取是否可以开启L1 Bank冲突优化功能。若可以开启该功能，则与TilingKey机制配合使用，通过增加TilingKey，关联Host侧与核函数（Kernel）侧实现，并在核函数（Kernel）侧增加代码实现分支，将MatmulConfig中的[enableL1BankConflictOptimise](../matmul_kernel/matmul_config.md#p84588523128)设置为true，即可优化L1上的Bank冲突。

## 函数原型

```
bool EnableL1BankConflictOptimise()
```

## 参数说明

无

## 返回值说明

-   false：核函数（Kernel）侧不能开启L1 Bank冲突优化。
-   true：核函数（Kernel）侧可以开启L1 Bank冲突优化。

## 约束说明

使用创建的Tiling对象调用本接口，且需在完成Tiling计算（[GetTiling](get_tiling.md)）后调用本接口。

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetShape(1024, 1024, 1024);
tiling.SetOrgShape(1024, 1024, 1024);
tiling.SetBias(true);
tiling.SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
// 核函数（Kernel）侧是否可以开启L1 Bank冲突优化，可与TilingKey机制结合使用
bool enableL1BankConflictOptimise = tiling.EnableL1BankConflictOptimise();
```

