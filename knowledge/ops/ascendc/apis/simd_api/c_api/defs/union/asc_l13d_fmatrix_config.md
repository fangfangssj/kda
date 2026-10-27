---
type: API
title: asc_l13d_fmatrix_config
description: asc_l13d_fmatrix_config用于设置asc_copy_l12l0a/asc_copy_l12l0b的3D格式搬运接口的Feature
  map属性参数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/union/asc_l13d_fmatrix_config.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/union.h
---

# asc_l13d_fmatrix_config

头文件路径为：`"c_api/defs/union.h"`。

asc_l13d_fmatrix_config用于设置[asc_copy_l12l0a](../../cube_datamove/cube_compute_load/index.md)/[asc_copy_l12l0b](../../cube_datamove/cube_compute_load/index.md)的3D格式搬运接口的Feature map属性参数。

## 结构体具体定义

```cpp
constexpr uint64_t ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE = 0;
union asc_l13d_fmatrix_config {
    uint64_t config = ASC_DEFAULT_L13D_FMATRIX_CONFIG_VALUE;
    struct {
        uint16_t l1_width;
        uint16_t l1_height;
        uint8_t padding_left_size;
        uint8_t padding_right_size;
        uint8_t padding_top_size;
        uint8_t padding_bottom_size;
    };
};
```

## 约束说明

`asc_l13d_fmatrix_config`类型的联合体变量被定义后必须对`l1_height`和`l1_width`进行赋值以符合实际场景及取值范围要求。

## 字段详解

**表1** 字段说明

|字段名|字段含义|
|----------|----------|
| l1_width | Feature map的width，取值范围：[1, 32767]。 |
| l1_height | Feature map的height，取值范围：[1, 32767]。 |
| padding_left_size | 左侧填充的大小，取值范围：[0, 255]。 |
| padding_right_size | 右侧填充的大小，取值范围：[0, 255]。 |
| padding_top_size | 上侧填充的大小，取值范围：[0, 255]。 |
| padding_bottom_size | 下侧填充的大小，取值范围：[0, 255]。 |
