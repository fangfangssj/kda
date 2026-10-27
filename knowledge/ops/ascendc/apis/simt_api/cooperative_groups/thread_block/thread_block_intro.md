---
type: API
title: thread_block简介
description: thread_block是对线程块的抽象，它代表了核函数的启动配置（启动的线程块个数，每个线程块内的线程数量等）。
tags:
- task.lookup
- topic.programming_model.cooperative_groups
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/cooperative_groups/thread_block/thread_block_intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# thread_block简介

`thread_block`是对线程块的抽象，它代表了核函数（Kernel）的启动配置（启动的线程块个数，每个线程块内的线程数量等）。

## Public成员函数

```cpp
static void sync();
static unsigned int thread_rank();
static dim3 group_index();
static dim3 thread_index();
static dim3 dim_threads();
static unsigned int num_threads();
static unsigned int size();
static dim3 group_dim();
```
