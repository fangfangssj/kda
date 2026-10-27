---
type: API
title: SetSparseIndex
description: 设置稀疏矩阵稠密化过程生成的索引矩阵。 索引矩阵在稠密化中的作用请参考MmadWithSparse。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetSparseIndex.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetSparseIndex

## 功能说明

设置稀疏矩阵[稠密化过程](../../../basic_api/cube_compute_isasi/mmad_compute/mmad_with_sparse.md#功能说明)生成的索引矩阵。

索引矩阵在稠密化中的作用请参考[MmadWithSparse](../../../basic_api/cube_compute_isasi/mmad_compute/mmad_with_sparse.md#section618mcpsimp)。

## 函数原型

```
__aicore__ inline void SetSparseIndex(const GlobalTensor<uint8_t>& indexGlobal)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| indexGlobal | 输入 | 索引矩阵在Global Memory上的首地址，类型为GlobalTensor。<br><br>索引矩阵的数据类型为uint2，需要由用户拼成uint8的数据类型，再传入本接口。索引矩阵的Format格式只支持NZ格式。 |

## 返回值说明

无

## 约束说明

-   索引矩阵的Format格式要求为NZ格式。
-   本接口仅支持在纯Cube模式（只有矩阵计算）且[MDL模板](matmul_config.md#mdl-template)的场景使用。

## 调用示例

```
#define ASCENDC_CUBE_ONLY // 开启纯Cube模式(只有矩阵计算)
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetSparseIndex(gm_index); //设置索引矩阵
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
