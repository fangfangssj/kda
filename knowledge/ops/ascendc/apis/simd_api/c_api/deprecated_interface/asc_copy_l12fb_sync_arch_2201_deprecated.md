---
type: API
title: asc_copy_l12fb_sync_arch_2201（废弃）
description: 头文件路径为："c_api/composite/cube_datamove_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_copy_l12fb_sync_arch_2201_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/cube_datamove_composite.h
---

# asc_copy_l12fb_sync_arch_2201（废弃）

## 功能说明

头文件路径为：`"c_api/composite/cube_datamove_composite.h"`。

**`asc_copy_l12fb_sync`接口已废弃，请使用[对应的非同步接口](../cube_datamove/cube_compute_load/asc_copy_l12fb_arch_2201.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将数据从L1 Buffer搬运到Fixpipe Buffer中，Fixpipe Buffer用于存放量化参数。

## 函数原型

- 同步搬运

    ```cpp
    __aicore__ inline void asc_copy_l12fb_sync(__fbuf__ void* dst, __cbuf__ void* src, uint32_t size)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数起始地址。 |
| src | 输入 | 源操作数起始地址。 |
| size | 输入 | 搬运数据大小（字节）。|

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- 当采用前n个数据搬运接口时，搬运数据大小要求32字节对齐。

## 调用示例

```cpp
// 同步接口按字节数连续搬运，此处搬运一个32字节数据块。
constexpr uint32_t copy_size = 32;
__cbuf__ half src[256];
__fbuf__ half dst[256];
asc_copy_l12fb_sync(dst, src, copy_size);
```
