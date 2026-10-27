---
type: API
title: GetArangeMaxMinTmpSize
description: 用于获取Arange Tiling参数：Arange接口能完成计算所需最大临时空间大小max和最小临时空间大小min。 由于Arange接口内部不需要用到临时空间，max和min均返回0。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.indexing
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/index_compute/GetArangeMaxMinTmpSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/index/arithprogression_tiling.h
---

# GetArangeMaxMinTmpSize

## 功能说明

用于获取Arange Tiling参数：Arange接口能完成计算所需最大临时空间大小max和最小临时空间大小min。

由于Arange接口内部不需要用到临时空间，max和min均返回0。

## 函数原型

> [!NOTE]说明
>GetArithProgressionMaxMinTmpSize接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetArangeMaxMinTmpSize接口。

```
void GetArangeMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

```
void GetArithProgressionMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| maxValue | 输出 | Arange接口能完成计算所需最大临时空间大小。<br> 说明：maxValue仅作为参考值，有可能大于UB剩余空间的大小，该场景下，开发者需要根据UB剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Arange接口能完成计算所需最小临时空间大小。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetArangeMaxMinTmpSize(maxValue, minValue);
```
