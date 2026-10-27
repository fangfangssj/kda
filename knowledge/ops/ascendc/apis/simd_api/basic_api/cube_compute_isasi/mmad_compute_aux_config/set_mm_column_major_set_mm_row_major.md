---
type: API
title: SetMMColumnMajor/SetMMRowMajor
description: 控制Mmad/MmadWithSparse优先通过M/N的哪个方向。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_aux_config/SetMMColumnMajor-SetMMRowMajor.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# SetMMColumnMajor/SetMMRowMajor

## 功能说明

控制Mmad/MmadWithSparse优先通过M/N的哪个方向。

## 函数原型

```cpp
__aicore__ inline void SetMMColumnMajor() // 代表CUBE将首先通过M方向，然后通过N方向生成结果
__aicore__ inline void SetMMRowMajor()   // 代表CUBE将首先通过N方向，然后通过M方向产生结果
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
AscendC::SetMMRowMajor();
```
