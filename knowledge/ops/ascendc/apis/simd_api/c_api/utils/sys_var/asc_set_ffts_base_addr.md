---
type: API
title: asc_set_ffts_base_addr
description: 在asc_sync_block_wait和asc_sync_block_arrive之前使用，设置核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取，并作为参数传入。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_set_ffts_base_addr.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
---

# asc_set_ffts_base_addr

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

在[asc_sync_block_wait](../../sync/inter_core_sync/asc_sync_block_wait.md)和[asc_sync_block_arrive](../../sync/inter_core_sync/asc_sync_block_arrive.md)之前使用，设置核间同步寄存器的基地址。需要在Host侧调用接口aclrtGetHardwareSyncAddr获取，并作为参数传入。

```cpp
aclError aclrtGetHardwareSyncAddr(void **addr);
```

## 函数原型

```cpp
__aicore__ inline void asc_set_ffts_base_addr(uint64_t config)
```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| config | 输入 | 核间同步的基地址。取值范围[0, 2^48-1]。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

- 在使用[asc_sync_block_wait](../../sync/inter_core_sync/asc_sync_block_wait.md)和[asc_sync_block_arrive](../../sync/inter_core_sync/asc_sync_block_arrive.md)之前必须使用本接口设置基地址。

## 调用示例

```cpp
// Host侧调用接口aclrtGetHardwareSyncAddr获取核间同步基地址ffts_addr
uint64_t config = *(__gm__ uint64_t*)ffts_addr;
asc_set_ffts_base_addr(config);
```
