---
type: API
title: GetTiling
description: 获取Mc2InitTiling参数和Mc2CcTiling参数。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Tiling/GetTiling.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# GetTiling

## 功能说明

获取[Mc2InitTiling](tiling_data_struct.md#table4835205712588)参数和[Mc2CcTiling](tiling_data_struct.md#table678914014562)参数。

## 函数原型

```
uint32_t GetTiling(::Mc2InitTiling& tiling)
```

```
uint32_t GetTiling(::Mc2CcTiling& tiling)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tiling | 输出 | Tiling结构体存储的Tiling信息。 |

## 返回值说明

-   返回值为0，则tiling计算成功，该Tiling结构体的值可以用于后续计算。
-   返回值非0，则tiling计算失败，该Tiling结果无法使用。

## 约束说明

无

## 调用示例

```
const char* groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling为算子组装的TilingData结构体，获取Mc2InitTiling
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling); // tiling为算子组装的TilingData结构体，获取Mc2CcTiling
```
