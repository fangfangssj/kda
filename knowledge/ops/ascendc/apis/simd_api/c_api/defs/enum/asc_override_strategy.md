---
type: API
title: asc_override_strategy
description: asc_override_strategy用于表示饱和控制策略，在调用asc_set_saturation_strategy和asc_get_saturation_strategy时使用。该策略影响CAST类全局饱和配置是否生效。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/enum/asc_override_strategy.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/enum.h
---

# asc_override_strategy

asc_override_strategy用于表示饱和控制策略，在调用[asc_set_saturation_strategy](../../spr/asc_set_saturation_strategy.md)和[asc_get_saturation_strategy](../../spr/asc_get_saturation_strategy.md)时使用。该策略影响CAST类全局饱和配置是否生效。

头文件路径为：`"c_api/defs/enum.h"`。

## 枚举类具体定义

```cpp
enum class asc_override_strategy : uint8_t {
    USE_API = 0,
    USE_GLOBAL = 1
};
```

## 枚举值详解

**表1** 枚举值说明

| 枚举值 | 说明 |
| ------ | ----------- |
| USE_API | 按单指令/API路径配置饱和。此时通过[asc_set_saturation_flag](../../spr/asc_set_saturation_flag.md)配置的全局`CAST`饱和不生效。 |
| USE_GLOBAL | 使用全局饱和配置。此时通过[asc_set_saturation_flag](../../spr/asc_set_saturation_flag.md)配置的`CAST`模式才会在精度转换中生效。 |

典型用法：`USE_GLOBAL`配合CAST饱和做float→int32动态量化，参见[asc_set_saturation_strategy](../../spr/asc_set_saturation_strategy.md)调用示例及[Cast样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/02_simd_c_api/03_c_api/02_reg_vector_compute/cast)；`USE_API`配合单指令饱和参见[Cast](../../../basic_api/reg_vector_compute/type_conversion/cast.md)中`SatMode`与饱和模式全局或单指令生效配置表。
