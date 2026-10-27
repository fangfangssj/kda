---
type: API
title: asc_get_rsvd_count
description: 此接口用于获取执行asc_reduce操作后剩余的元素数量。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_reduce/asc_get_rsvd_count.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_reduce.h
---

# asc_get_rsvd_count

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_reduce.h"`。

此接口用于获取执行[asc_reduce](asc_reduce.md)操作后剩余的元素数量。

## 函数原型

```cpp
  __aicore__ inline int64_t asc_get_rsvd_count()
```

## 参数说明

无

## 返回值说明

执行[asc_reduce](asc_reduce.md)操作后剩余的元素数量。

## 流水类型

PIPE_S

## 约束说明

- 需和[asc_reduce](asc_reduce.md)操作配合使用。
- 需通过同步操作确保[asc_reduce](asc_reduce.md)执行完成后再调用本接口获取结果。

## 调用示例

```cpp
// 初始化dst、src0、src1和total_length(参与计算的数据长度)
int total_length = 256;
__ubuf__ uint32_t src0[total_length];
__ubuf__ uint32_t src1[total_length];
__ubuf__ uint32_t dst[total_length];
asc_reduce(dst, src0, src1, total_length);
asc_sync_notify(PIPE_V, PIPE_S, 0); // 设置等待和同步信号
asc_sync_wait(PIPE_V, PIPE_S, 0);
int64_t result = asc_get_rsvd_count();  // 获取结果
```
