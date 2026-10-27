---
type: API
title: asc_position_quarter_mode
description: asc_position_quarter_mode用于表示源或目的操作数的四分之一位置，供Reg类型转换接口选择数据位置。对应的ASC_DISPERSE_*_QUARTER常量可直接作为接口参数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/enum/asc_position_quarter_mode.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/enum.h
---

# asc_position_quarter_mode

`asc_position_quarter_mode`用于表示源或目的操作数的四分之一位置，供Reg类型转换接口选择数据位置。对应的`ASC_DISPERSE_*_QUARTER`常量可直接作为接口参数。

头文件路径为：`"c_api/defs/enum.h"`。

## 枚举类具体定义

```cpp
enum class asc_position_quarter_mode {
    DISPERSE_FIRST_QUARTER = 0,
    DISPERSE_SECOND_QUARTER,
    DISPERSE_THIRD_QUARTER,
    DISPERSE_FOURTH_QUARTER
};
```

## 枚举值详解

**表1** 枚举值说明

| 枚举值 | 取值 | 说明 | 对应常量 |
| --- | --- | --- | --- |
| DISPERSE_FIRST_QUARTER | 0 | 选择第一个四分之一位置。 | ASC_DISPERSE_FIRST_QUARTER |
| DISPERSE_SECOND_QUARTER | 1 | 选择第二个四分之一位置。 | ASC_DISPERSE_SECOND_QUARTER |
| DISPERSE_THIRD_QUARTER | 2 | 选择第三个四分之一位置。 | ASC_DISPERSE_THIRD_QUARTER |
| DISPERSE_FOURTH_QUARTER | 3 | 选择第四个四分之一位置。 | ASC_DISPERSE_FOURTH_QUARTER |
