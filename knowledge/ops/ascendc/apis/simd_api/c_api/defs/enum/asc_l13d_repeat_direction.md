---
type: API
title: asc_l13d_repeat_direction
description: asc_l13d_repeat_direction用于表示3D img2col搬运的repeat迭代方向，作为asc_set_l13d_rpt和asc_set_l13d_rpt_b接口的repeat_direction参数类型。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/enum/asc_l13d_repeat_direction.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/c_api/defs/enum.h
---

# asc_l13d_repeat_direction

`asc_l13d_repeat_direction`用于表示3D img2col搬运的repeat迭代方向，作为[asc_set_l13d_rpt](../../cube_datamove/cube_load_aux_config/asc_set_l13d_rpt.md)和[asc_set_l13d_rpt_b](../../cube_datamove/cube_load_aux_config/asc_set_l13d_rpt_b.md)接口的`repeat_direction`参数类型。

头文件路径为：`"c_api/defs/enum.h"`。

## 枚举类具体定义

```cpp
enum class asc_l13d_repeat_direction : uint8_t {
    M_DIRECTION = 0,
    K_DIRECTION = 1
};
```

## 枚举值详解

**表1** 枚举值说明

| 枚举值 | 取值 | repeat迭代方向 | `repeat_stride`单位 |
| --- | --- | --- | --- |
| M_DIRECTION | 0 | 沿height（M）方向迭代。 | 16个元素。 |
| K_DIRECTION | 1 | 沿width（K）方向迭代。 | `32 / sizeof(data_type)`个元素。其中，`data_type`为3D搬运接口源操作数的数据类型。 |
