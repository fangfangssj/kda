---
type: API
title: global_tensor::SetL2CacheHint（废弃）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/deprecated_interface/set_l2_cache_hint_deprecated.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# global_tensor::SetL2CacheHint（废弃）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

**该接口已废弃，请使用[global_tensor::set_l2_cache_hint](../tensor/tensor.md#set_l2_cache_hint)替代。**

设置Global Tensor访问Global Memory时使用的L2 Cache Hint。迁移时需要将PascalCase的`AscendC::Te::CacheMode`枚举值替换为snake_case的`cache_mode`枚举值。

## 函数原型

```cpp
__aicore__ inline constexpr void SetL2CacheHint(AscendC::Te::CacheMode mode);
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| mode | 输入 | L2 Cache模式。旧接口使用`AscendC::Te::CacheMode`，替代接口使用`cache_mode`。 |

## 返回值说明

无。

## 流水类型

无

## 约束说明

该接口仅适用于`global_tensor`，不适用于`local_tensor`。

## 调用示例

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

gm_tensor.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE); // 废弃写法
gm_tensor.set_l2_cache_hint(cache_mode::disable);                      // 替代写法
```
