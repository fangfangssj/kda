---
type: API
title: WaitGetTensorC
description: 当使用GetTensorC异步接口将结果矩阵从GM拷贝到UB，且UB后续需要进行Vector计算时，需要调用WaitGetTensorC进行同步。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/WaitGetTensorC.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/adv_api/matmul/matmul_client.h
---

# WaitGetTensorC

## 功能说明

当使用GetTensorC异步接口将结果矩阵从GM拷贝到Unified Buffer（UB），且UB后续需要进行Vector计算时，需要调用WaitGetTensorC进行同步。

## 函数原型

```
__aicore__ inline void WaitGetTensorC()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
// 异步模式样例
mm.template Iterate<false>();
// 其他操作
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {
    mm.template GetTensorC<false>(ubCmatrix);
    mm.WaitGetTensorC();
    // Vector操作
}
```
