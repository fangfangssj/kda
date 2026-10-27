---
type: API
title: GetLength
description: 获取LocalTensor数据长度。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetLength.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tensor.h
---

# GetLength<a name="ZH-CN_TOPIC_0000002364805853"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取LocalTensor数据长度。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline uint32_t GetLength() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor数据长度，单位为字节。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 获取localTensor的长度(单位为Byte)，数据类型为int32_t，所以是16*sizeof(int32_t)
uint32_t len = inputLocal.GetLength();
// inputLocal:0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// len: 64
```

