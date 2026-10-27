---
type: API
title: ASC_VDEQ_SIZE
description: ASC_VDEQ_SIZE表示asc_set_deq_scale接口中scale_arr、offset_arr和sign_mode_arr数组的元素个数，值为16。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/constant/asc_vdeq_size.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# ASC_VDEQ_SIZE

ASC_VDEQ_SIZE表示[asc_set_deq_scale](../../vector_compute/vector_fused/asc_set_deq_scale.md)接口中scale_arr、offset_arr和sign_mode_arr数组的元素个数，值为16。

头文件路径：`c_api/defs/constant.h`。

## 常量定义

```cpp
constexpr uint8_t ASC_VDEQ_SIZE = 16;
```
