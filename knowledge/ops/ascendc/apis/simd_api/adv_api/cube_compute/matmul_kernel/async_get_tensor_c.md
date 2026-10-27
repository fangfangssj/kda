---
type: API
title: AsyncGetTensorC
description: 获取Iterate接口异步计算的结果矩阵。该接口功能已被GetTensorC覆盖，建议直接使用GetTensorC异步接口。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/AsyncGetTensorC.md
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

# AsyncGetTensorC

## 功能说明

获取Iterate接口异步计算的结果矩阵。该接口功能已被[GetTensorC](get_tensor_c.md)覆盖，建议直接使用GetTensorC异步接口。

## 函数原型

```
__aicore__ inline void AsyncGetTensorC(const LocalTensor<DstT>& c)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| c | 输出 | 结果矩阵 |

## 返回值说明

无

## 约束说明

当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。
