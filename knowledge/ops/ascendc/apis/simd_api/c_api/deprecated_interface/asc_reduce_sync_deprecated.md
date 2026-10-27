---
type: API
title: asc_reduce_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_reduce_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_reduce_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_reduce_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_reduce/asc_reduce.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

以内置固定模式对应的二进制或者用户自定义输入的数值对应的gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。

## 函数原型

- 同步计算

  ```cpp
  __aicore__ inline void asc_reduce_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1, uint32_t count)
  __aicore__ inline void asc_reduce_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1, uint32_t count)
  ```

## 参数说明

**表1** 参数说明

| 参数名                | 输入/输出 | 描述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:-------------------| :-----|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dst                | 输出 | 目的操作数（矢量）的起始地址。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| src0               | 输入 | 源操作数（矢量）的起始地址。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| src1               | 输入 | 在用户自定义模式下，由用户自定义的数据收集的掩码。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| count              | 输入 | 参与计算的元素个数。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 接口不支持通过[asc_set_vector_mask](../vector_compute/vector_mask_config/asc_set_vector_mask.md)配置掩码。
- dst、src0、src1的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

```cpp
//total_length指参与计算的数据总长度
constexpr uint64_t total_length = 128;
__ubuf__ uint16_t src0[total_length];
__ubuf__ uint16_t src1[total_length/16];
__ubuf__ uint16_t dst[total_length];
asc_reduce_sync(dst, src0, src1, total_length);
```
