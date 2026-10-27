---
type: API
title: asc_fill_value_config
description: asc_fill_value_config为fill_value的初始化参数结构体，包含asc_fill_l0a/asc_fill_l0b/asc_fill_l1接口需要配置的各种初始化参数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/union/asc_fill_value_config.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/union.h
---

# asc_fill_value_config

头文件路径为：`"c_api/defs/union.h"`。

asc_fill_value_config为fill_value的初始化参数结构体，包含[asc_fill_l0a](../../cube_datamove/cube_load_aux_config/asc_fill_l0a.md)/[asc_fill_l0b](../../cube_datamove/cube_load_aux_config/asc_fill_l0b.md)/[asc_fill_l1](../../cube_datamove/cube_load_aux_config/asc_fill_l1.md)接口需要配置的各种初始化参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_FILL_VALUE_DEFAULT_VALUE = 0x0000000000000000;
union asc_fill_value_config {
    uint64_t config = ASC_FILL_VALUE_DEFAULT_VALUE;
    struct {
        uint64_t repeat : 15;
        uint64_t reserved1 : 1;
        uint64_t blk_num : 15;
        uint64_t reserved2 : 1;
        uint64_t dst_gap : 15;
        uint64_t reserved3 : 17;
    };
};
```

## 字段详解

**表1** 字段说明

|字段名|字段含义|
|----------|----------|
| repeat | 迭代次数，默认值：0。取值范围：[0, 32767]。 |
| blk_num | 每次迭代初始化的数据块个数，默认值：0。取值范围：[0, 32767]。<br>目的操作数位于L1 Buffer时，数据块大小是32B；目的操作数位于L0A Buffer/L0B Buffer时，数据块大小是512B。 |
| dst_gap | 目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离，默认值：0。取值范围：[0, 32767]。<br>目的操作数位于L1 Buffer时，单位是32B；目的操作数位于L0A Buffer/L0B Buffer时，单位是512B。 |