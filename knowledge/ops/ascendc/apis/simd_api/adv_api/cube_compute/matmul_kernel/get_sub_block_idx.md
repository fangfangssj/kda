---
type: API
title: GetSubBlockIdx
description: 获取当前AIV核在当前AI Core分组中的ID，需在REGIST_MATMUL_OBJ之后调用。仅分离架构支持。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetSubBlockIdx.md
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

# GetSubBlockIdx

## 功能说明

获取当前AIV核的ID。分离架构下，一个AI Core由Cube Core（AIC）和Vector Core（AIV）按照一定比例1：N进行组合，其中N个AIV的子核ID分别为0, 1, ..., N-1。

Matmul::GetSubBlockIdx\(\)与基础API接口[AscendC::GetSubBlockIdx\(\)](../../../basic_api/tool_interface/system_resources_and_variables/get_sub_block_idx_isasi.md)的区别在于，Matmul::GetSubBlockIdx\(\)用于获取当前AIV核在当前AI Core分组中的ID，而AscendC::GetSubBlockIdx\(\)获取的是AIV核在所有AI Core分组中的逻辑ID。例如，有10组AI Core，AIC与AIV的比例为1:2，共20个AIV核。调用Matmul::GetSubBlockIdx\(\)时，20个AIV的获取结果依次为0, 1, 0, 1, 0, 1, ..., 0, 1。调用AscendC::GetSubBlockIdx\(\)时，20个AIV的获取结果依次为0, 1, 2, 3, 4, 5, ..., 18, 19。

## 函数原型

```
__aicore__  inline uint8_t GetSubBlockIdx()
```

## 参数说明

无

## 返回值说明

当前AIV核的ID。

## 约束说明

-   该接口仅支持在分离架构下使用，否则返回随机值。
-   在分离架构中，AIV核的ID会在[REGIST\_MATMUL\_OBJ\(\)](regist_matmul_obj.md)接口内部自动初始化和赋值。因此，需要在调用REGIST\_MATMUL\_OBJ\(\)接口之后，再调用本接口，以获取正确的ID。
-   若在算子程序中调用[SetSubBlockIdx\(\)](set_sub_block_idx.md)后， GetSubBlockIdx\(\)接口将返回由SetSubBlockIdx接口设置的ID值。

## 调用示例

```
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;

AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.GetSubBlockIdx(); // 获取子核ID
```
