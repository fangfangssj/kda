---
type: API
title: WaitIterateBatch
description: 等待IterateBatch异步接口或IterateNBatch异步接口返回，支持连续输出到Global Memory。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/WaitIterateBatch.md
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

# WaitIterateBatch

## 功能说明

等待[IterateBatch](iterate_batch.md)异步接口或[IterateNBatch](iterate_n_batch.md)异步接口返回，支持连续输出到Global Memory。

## 函数原型

```
__aicore__ inline void WaitIterateBatch()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

-   配套[IterateBatch](iterate_batch.md)或[IterateNBatch](iterate_n_batch.md)异步接口使用。
-   仅支持连续输出至Global Memory。
-   当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
AscendC::Matmul<aType, bType, cType, biasType> mm;
mm.SetTensorA(gm_a[offsetA]);
mm.SetTensorB(gm_b[offsetB]);
if (tiling.isBias) {
    mm.SetBias(gm_bias[offsetBias]);
}
mm.IterateBatch(gm_c[offsetC], batchA, batchB, false);
// do some other compute tasks
mm.WaitIterateBatch(); // 等待IterateBatch完成
```
