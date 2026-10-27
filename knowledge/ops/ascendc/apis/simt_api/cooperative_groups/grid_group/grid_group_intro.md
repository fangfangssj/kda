---
type: API
title: grid_group简介
description: grid_group是对一个核函数启动的所有SIMT线程的抽象，线程范围覆盖当前Grid中的所有线程块及其线程。通过grid_group可以查询Grid维度、线程块排名、全局线程排名，并通过sync()在Grid范围内进行跨线程块同步。
tags:
- task.lookup
- topic.programming_model.cooperative_groups
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/grid_group/grid_group_intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
source_headers:
- include/simt_api/cooperative_groups.h
---

# grid_group简介

`grid_group`是对一个核函数（Kernel）启动的所有SIMT线程的抽象，线程范围覆盖当前Grid中的所有线程块及其线程。通过`grid_group`可以查询Grid维度、线程块排名、全局线程排名，并通过`sync()`在Grid范围内进行跨线程块同步。

## Public成员函数

```cpp
bool is_valid() const;
void sync() const;
static unsigned long long thread_rank();
static unsigned long long block_rank();
static unsigned long long num_threads();
static unsigned long long size();
static unsigned long long num_blocks();
static dim3 dim_blocks();
static dim3 group_dim();
static dim3 block_index();
```

## 约束说明

`grid_group`仅支持SIMT编程场景，不支持SIMD与SIMT混合编程场景。
