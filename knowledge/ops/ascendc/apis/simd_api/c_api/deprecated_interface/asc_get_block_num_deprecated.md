---
type: API
title: asc_get_block_num（废弃）
description: 该接口已废弃，请使用内置变量block_num替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_get_block_num_deprecated.md
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

# asc_get_block_num（废弃）

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

**该接口已废弃，请使用[内置变量block_num](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)替代。**

获取当前任务配置的核数，用于代码内部的多核逻辑控制等。建议在代码中直接使用内置变量`block_num`：

```cpp
uint32_t blockLength = C_API_TOTAL_LENGTH / block_num;  // 替代 asc_get_block_num()
```

### 迁移说明

内置变量`block_num`的值为当前任务配置的核数，与原接口返回值一致，替代时直接使用即可。使用前请参考[内置变量说明](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#内置变量)确认适用场景。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_block_num()
```

## 参数说明

无

## 返回值说明

当前任务配置的核数。

## 流水类型

`PIPE_S`

## 约束说明

无

## 调用示例

```cpp
uint32_t blockLength = C_API_TOTAL_LENGTH / block_num;  // 获取当前任务配置的核数，替代 asc_get_block_num()
```