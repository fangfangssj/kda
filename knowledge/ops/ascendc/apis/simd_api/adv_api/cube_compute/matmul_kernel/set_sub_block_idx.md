---
type: API
title: SetSubBlockIdx
description: 设置当前AIV核的ID。分离架构下，一个AI Core由Cube Core（AIC）和Vector Core（AIV）按照一定比例1：N进行组合，其中N个AIV核的ID分别为0,
  1, ..., N-1。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetSubBlockIdx.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/adv_api/matmul/matmul.h
---

# SetSubBlockIdx

## 功能说明

设置当前AIV核的ID。分离架构下，一个AI Core由Cube Core（AIC）和Vector Core（AIV）按照一定比例1：N进行组合，其中N个AIV核的ID分别为0, 1, ..., N-1。

## 函数原型

```
__aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| subBlockIdx | 输入 | 当前AIV核的ID。 |

## 返回值说明

无

## 约束说明

-   该接口仅支持在分离架构下使用。
-   在分离架构中，AIV核的ID会在[REGIST\_MATMUL\_OBJ\(\)](regist_matmul_obj.md)接口内部自动初始化和赋值。如果在算子程序中使用了REGIST\_MATMUL\_OBJ\(\)接口，则不建议调用此接口；若未使用REGIST\_MATMUL\_OBJ\(\)接口，则请调用此接口并将子核ID设置为0。

## 调用示例

```
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;

Matmul<aType, bType, cType, biasType, CFG_NORM> mm;
mm.SetSubBlockIdx(0); // 子核ID设置为0
```
