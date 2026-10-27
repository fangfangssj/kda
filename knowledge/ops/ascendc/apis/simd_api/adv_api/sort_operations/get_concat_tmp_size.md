---
type: API
title: GetConcatTmpSize
description: 获取Concat接口所需的临时空间大小，单位为字节。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.ordering
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/sort_operations/GetConcatTmpSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/adv_api/sort/sort_tiling_intf.h
---

# GetConcatTmpSize

## 功能说明

获取Concat接口所需的临时空间大小，单位为字节。

## 函数原型

```
uint32_t GetConcatTmpSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t elemCount, const uint32_t dataTypeSize)
```

## 参数说明

**表1**  接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见[构造及析构函数](../../../utils_api/platform_info/platform_ascend_c/constructor_and_destructor.md)。 |
| elemCount | 输入 | 输入元素个数。 |
| dataTypeSize | 输入 | 输入数据大小（单位为字节）。 |

## 返回值说明

Concat接口所需的临时空间大小的字节数。

## 约束说明

无

## 调用示例

```
fe::PlatFormInfos platform_info;
auto plat = platform_ascendc::PlatformAscendC(&platform_info);
const uint32_t elemCount = 128;
AscendC::GetConcatTmpSize(plat, elemCount, 2);
```

- [Sort 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/sort_950.md)
- [TopK 950 Direct Invoke 算子知识](../../../../operators/direct/math/math/top_k_950.md)
