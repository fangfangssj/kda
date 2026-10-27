---
type: API
title: asc_load_dev
description: 不经过DCache直接从GM地址读取整型数据。 当多核操作GM地址且数据无法对齐到Cache Line时，经过DCache读写可能引入Cache
  Line粒度的数据覆盖。此时，可使用本接口绕过DCache读取GM数据。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/scalar_compute/scalar_load/asc_load_dev.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/scalar_compute/scalar_load.h
---

# asc_load_dev

## 功能说明

头文件路径为：`"c_api/scalar_compute/scalar_load.h"`。

不经过DCache直接从GM地址读取整型数据。
当多核操作GM地址且数据无法对齐到Cache Line时，经过DCache读写可能引入Cache Line粒度的数据覆盖。此时，可使用本接口绕过DCache读取GM数据。

## 函数原型

```cpp
__aicore__ inline int8_t asc_load_dev(__gm__ int8_t* addr)

__aicore__ inline uint8_t asc_load_dev(__gm__ uint8_t* addr)

__aicore__ inline int16_t asc_load_dev(__gm__ int16_t* addr)

__aicore__ inline uint16_t asc_load_dev(__gm__ uint16_t* addr)

__aicore__ inline int32_t asc_load_dev(__gm__ int32_t* addr)

__aicore__ inline uint32_t asc_load_dev(__gm__ uint32_t* addr)

__aicore__ inline int64_t asc_load_dev(__gm__ int64_t* addr)

__aicore__ inline uint64_t asc_load_dev(__gm__ uint64_t* addr)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| addr | 输入 | 源GM地址。支持的数据类型为`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`。 |

## 返回值说明

从GM读取的数据，返回值的数据类型与`addr`指向的数据类型一致。

## 流水类型

PIPE_S

## 约束说明

仅支持整型数据，不支持浮点类型。

## 调用示例

```cpp
// addr是外部输入的GM地址，类型为__gm__ int32_t*。
int32_t value = asc_load_dev(addr);
```
