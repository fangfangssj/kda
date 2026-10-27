---
type: API
title: asc_get_block_idx（废弃）
description: 该接口已废弃，请使用内置变量block_idx结合asc_get_sub_block_num和asc_get_sub_block_id接口替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_get_block_idx_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
---

# asc_get_block_idx（废弃）

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

**该接口已废弃，请使用[内置变量block_idx](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)结合[asc_get_sub_block_num](../utils/sys_var/asc_get_sub_block_num.md)和[asc_get_sub_block_id](../utils/sys_var/asc_get_sub_block_id.md)接口替代。**

获取当前运行核（逻辑核）的索引。替代时需按逻辑位置计算：

```cpp
int64_t blockIdx = block_idx * asc_get_sub_block_num() + asc_get_sub_block_id();  // 替代 asc_get_block_idx()
__gm__ float* xGm = x + blockIdx * blockLength;
```

### 迁移说明

- 在非Mix场景下，可直接使用内置变量`block_idx`，其值等于`asc_get_block_idx()`的返回值。
- 在Mix场景（使用`__mix__`函数执行空间限定符）下，内置变量`block_idx`仅标识当前组合在整个grid中的位置，逻辑核索引需结合子块信息计算：`logic_idx = block_idx * sub_block_num + sub_block_id`，即`block_idx * asc_get_sub_block_num() + asc_get_sub_block_id()`；该表达式的值与原接口返回值一致。
- 使用前请参考[内置变量说明](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)确认适用场景。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_block_idx()
```

## 参数说明

无

## 返回值说明

当前运行核的索引。

## 流水类型

`PIPE_S`

## 约束说明

无

## 调用示例

```cpp
int64_t blockIdx = block_idx * asc_get_sub_block_num() + asc_get_sub_block_id();  // 获取当前运行核的索引，替代 asc_get_block_idx()
```
