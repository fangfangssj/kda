---
type: API
title: Tensor
description: Tensor由Pointer和Layout组成。Pointer描述数据地址及存储位置，Layout描述数据的Shape和Stride。开发者可以通过Tensor统一访问Global
  Memory或片上Buffer中的数据，并通过切片创建共享底层存储的子Tensor。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/tensor/tensor_structure.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/tensor_api/tensor/tensor.h
---

# Tensor

Tensor由Pointer和Layout组成。Pointer描述数据地址及存储位置，Layout描述数据的Shape和Stride。开发者可以通过Tensor统一访问Global Memory或片上Buffer中的数据，并通过切片创建共享底层存储的子Tensor。

## Tensor相关数据结构

- **[Tensor](tensor.md)**
- **[Pointer](pointer.md)**
- **[view_engine](view_engine.md)**

## Tensor相关接口

- **[make_tensor](make_tensor.md)**
- **[make_mem_ptr](make_mem_ptr.md)**
- **[slice](slice.md)**
