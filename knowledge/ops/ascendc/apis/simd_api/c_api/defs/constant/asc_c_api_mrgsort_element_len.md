---
type: API
title: ASC_C_API_MRGSORT_ELEMENT_LEN
description: ASC_C_API_MRGSORT_ELEMENT_LEN表示MrgSort4操作支持的队列数量，值为4。该常量用于asc_mrgsort4接口的src数组长度，以及asc_get_vms4_sr接口的sorted_num数组长度。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/defs/constant/asc_c_api_mrgsort_element_len.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# ASC_C_API_MRGSORT_ELEMENT_LEN

ASC_C_API_MRGSORT_ELEMENT_LEN表示MrgSort4操作支持的队列数量，值为4。该常量用于[asc_mrgsort4](../../vector_compute/vector_sort/asc_mrgsort4.md)接口的src数组长度，以及[asc_get_vms4_sr](../../vector_compute/vector_sort/asc_get_vms4_sr.md)接口的sorted_num数组长度。

头文件路径：`c_api/defs/constant.h`。

## 常量定义

```cpp
constexpr uint8_t ASC_C_API_MRGSORT_ELEMENT_LEN = 4;
```
